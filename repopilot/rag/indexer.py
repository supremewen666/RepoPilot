"""Compatibility indexing helpers layered on top of the EasyRAG orchestrator."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

from repopilot.compat import Document
from repopilot.config import get_rag_index_path, get_rag_working_dir, get_rag_workspace, get_repo_root
from repopilot.rag.easyrag import EasyRAG
from repopilot.rag.operate import ChunkingConfig, chunk_documents, load_repo_documents

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _run_async(awaitable: object) -> object:
    """Run an async EasyRAG call from synchronous compatibility helpers."""

    try:
        return asyncio.run(awaitable)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(awaitable)
        finally:
            loop.close()


def _resolve_legacy_storage_location() -> tuple[Path, str]:
    """Map the legacy index path env var onto a working dir and workspace."""

    if "REPOPILOT_RAG_WORKING_DIR" in os.environ or "REPOPILOT_RAG_WORKSPACE" in os.environ:
        return get_rag_working_dir(), get_rag_workspace()

    legacy_index_path = get_rag_index_path()
    if "REPOPILOT_RAG_INDEX_PATH" in os.environ:
        return legacy_index_path.parent, legacy_index_path.stem
    return get_rag_working_dir(), get_rag_workspace()


def _tokenize(text: str) -> list[str]:
    """Tokenize text for the legacy JSON snapshot."""

    return _TOKEN_PATTERN.findall(text.lower())


async def _build_workspace(documents: list[Document], working_dir: Path, workspace: str) -> None:
    """Populate one EasyRAG workspace in a single async lifecycle."""

    rag = EasyRAG(working_dir=working_dir, workspace=workspace)
    await rag.initialize_storages()
    try:
        await rag.ainsert_documents(documents)
    finally:
        await rag.finalize_storages()


def build_vector_index(documents: list[Document]) -> None:
    """
    Build the EasyRAG workspace and a legacy JSON snapshot for compatibility.

    The legacy JSON file remains useful for tests and local inspection, while
    the real retrieval state now lives under the EasyRAG working directory.
    """

    working_dir, workspace = _resolve_legacy_storage_location()
    _run_async(_build_workspace(documents, working_dir, workspace))

    payload = [
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
            "tokens": _tokenize(chunk.page_content),
        }
        for chunk in chunk_documents(documents, config=ChunkingConfig())
    ]
    index_path = get_rag_index_path()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def rebuild_document_index(repo_root: str | Path | None = None) -> Path:
    """Discover repository docs and rebuild the default EasyRAG workspace."""

    root = Path(repo_root).resolve() if repo_root is not None else get_repo_root()
    documents = load_repo_documents(root)
    build_vector_index(documents)
    return get_rag_working_dir() / get_rag_workspace()
