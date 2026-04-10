"""Task status storage backends for service-managed RAG indexing."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from repopilot.config import get_rag_storage_backend, get_task_status_path
from repopilot.rag.storage.base import BaseTaskStatusStorage
from repopilot.rag.storage.production import PostgresTaskStatusStorage


def _read_json(path: Path, default: Any) -> Any:
    """Read JSON data if the file exists, otherwise return a default value."""

    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _write_json(path: Path, payload: Any) -> None:
    """Persist JSON using UTF-8 and stable pretty printing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class JSONTaskStatusStorage(BaseTaskStatusStorage):
    """Persist index task state as one local JSON document."""

    def __init__(self, working_dir: str, workspace: str) -> None:
        super().__init__(working_dir, workspace)
        self._path = get_task_status_path()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    async def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._tasks = _read_json(self._path, {})

    async def finalize(self) -> None:
        with self._lock:
            _write_json(self._path, self._tasks)

    async def upsert_task(self, task: dict[str, Any]) -> None:
        task_id = str(task["task_id"])
        with self._lock:
            self._tasks[task_id] = dict(task)
            _write_json(self._path, self._tasks)

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return dict(task) if task is not None else None

    async def list_tasks(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            values = sorted(
                (dict(item) for item in self._tasks.values()),
                key=lambda item: str(item.get("requested_at", "")),
                reverse=True,
            )
        return values[: max(limit, 0)]


def resolve_task_status_storage_cls(backend_name: str | None = None) -> type[Any]:
    """Resolve the task status storage class aligned with the configured backend."""

    normalized = (backend_name or get_rag_storage_backend()).strip().lower()
    if normalized == "postgres_qdrant":
        return PostgresTaskStatusStorage
    return JSONTaskStatusStorage
