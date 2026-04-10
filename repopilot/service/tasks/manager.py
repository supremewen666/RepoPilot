"""In-process index task orchestration for the service layer."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from repopilot.config import get_rag_working_dir, get_rag_workspace, get_repo_root
from repopilot.rag import BaseTaskStatusStorage, EasyRAG
from repopilot.service.tasks.storage import resolve_task_status_storage_cls

_TASK_STORAGE_WORKSPACE = "__rag_tasks__"


def _utc_now() -> str:
    """Return one UTC timestamp formatted for JSON status payloads."""

    return datetime.now(timezone.utc).isoformat()


class IndexTaskManager:
    """Manage in-process background index tasks with persistent status."""

    def __init__(
        self,
        *,
        working_dir: str | Path | None = None,
        default_workspace: str | None = None,
        task_status_storage_cls: type[BaseTaskStatusStorage] | None = None,
    ) -> None:
        self.working_dir = Path(working_dir or get_rag_working_dir()).resolve()
        self.default_workspace = default_workspace or get_rag_workspace()
        storage_cls = task_status_storage_cls or resolve_task_status_storage_cls()
        self.task_status_storage = storage_cls(str(self.working_dir), _TASK_STORAGE_WORKSPACE)
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize persistent task status storage."""

        if self._initialized:
            return
        await self.task_status_storage.initialize()
        self._initialized = True

    async def finalize(self) -> None:
        """Flush task status storage and wait for in-flight tasks."""

        if not self._initialized:
            return
        if self._background_tasks:
            await asyncio.gather(*list(self._background_tasks), return_exceptions=True)
        await self.task_status_storage.finalize()
        self._initialized = False

    async def create_task(self, *, action: str, workspace: str | None = None, doc_ids: list[str] | None = None) -> dict[str, Any]:
        """Create and schedule one index maintenance task."""

        normalized_action = action.strip().lower()
        normalized_doc_ids = list(dict.fromkeys(str(doc_id).strip() for doc_id in list(doc_ids or []) if str(doc_id).strip()))
        if normalized_action in {"rebuild_docs", "delete_docs"} and not normalized_doc_ids:
            raise ValueError(f"`doc_ids` are required for {normalized_action}.")

        task = {
            "task_id": f"task::{uuid4().hex}",
            "action": normalized_action,
            "workspace": workspace or self.default_workspace,
            "doc_ids": normalized_doc_ids,
            "status": "queued",
            "requested_at": _utc_now(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }
        await self.task_status_storage.upsert_task(task)
        background = asyncio.create_task(self._run_task(dict(task)))
        self._background_tasks.add(background)
        background.add_done_callback(self._background_tasks.discard)
        return task

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Return one persisted task record."""

        return await self.task_status_storage.get_task(task_id)

    async def list_tasks(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """List recent persisted task records."""

        return await self.task_status_storage.list_tasks(limit=limit)

    async def _run_task(self, task: dict[str, Any]) -> None:
        """Execute one index maintenance task and persist status transitions."""

        task["status"] = "running"
        task["started_at"] = _utc_now()
        await self.task_status_storage.upsert_task(task)
        try:
            task["result"] = await self._execute_action(
                action=str(task["action"]),
                workspace=str(task["workspace"]),
                doc_ids=list(task.get("doc_ids", []) or []),
            )
            task["status"] = "succeeded"
        except Exception as exc:
            task["status"] = "failed"
            task["error"] = str(exc)
        finally:
            task["finished_at"] = _utc_now()
            await self.task_status_storage.upsert_task(task)

    async def _execute_action(self, *, action: str, workspace: str, doc_ids: list[str]) -> dict[str, Any]:
        """Run the requested index action in one EasyRAG lifecycle."""

        rag = EasyRAG(working_dir=self.working_dir, workspace=workspace)
        await rag.initialize_storages()
        try:
            if action == "delete_docs":
                deleted = await rag.adelete_documents(doc_ids)
                stats = await rag.get_stats()
                return {"deleted": deleted, "stats": stats}

            documents = EasyRAG.load_repo_documents(get_repo_root())
            current_doc_ids = {str(document.metadata.get("doc_id", "")).strip() for document in documents}
            selected_documents = [document for document in documents if str(document.metadata.get("doc_id", "")).strip() in set(doc_ids)] if doc_ids else documents

            if action == "full_sync":
                indexed_doc_ids = {
                    str(status.get("document_id", "")).strip()
                    for status in await rag.doc_status_storage.list_statuses()
                    if str(status.get("document_id", "")).strip()
                }
                stale_doc_ids = sorted(indexed_doc_ids - current_doc_ids)
                if stale_doc_ids:
                    await rag.adelete_documents(stale_doc_ids)
                indexed = await rag.ainsert_documents(selected_documents) if selected_documents else {"documents": 0, "chunks": 0, "entities": 0, "relations": 0, "pdf_documents": 0}
                return {"indexed": indexed, "stats": await rag.get_stats(), "stale_doc_ids": stale_doc_ids}

            if action == "rebuild_docs":
                missing_doc_ids = sorted(set(doc_ids) - current_doc_ids)
                if missing_doc_ids:
                    await rag.adelete_documents(missing_doc_ids)
                indexed = await rag.ainsert_documents(selected_documents) if selected_documents else {"documents": 0, "chunks": 0, "entities": 0, "relations": 0, "pdf_documents": 0}
                return {"indexed": indexed, "stats": await rag.get_stats(), "deleted_missing_doc_ids": missing_doc_ids}

            raise ValueError(f"Unsupported task action: {action}")
        finally:
            await rag.finalize_storages()
