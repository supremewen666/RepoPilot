"""Compatibility retrieval helpers layered on top of EasyRAG."""

from __future__ import annotations

import asyncio

from repopilot.compat import Document
from repopilot.config import get_rag_working_dir, get_rag_workspace
from repopilot.rag.base import QueryParam
from repopilot.rag.easyrag import EasyRAG


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


async def _search(query: str, k: int) -> list[Document]:
    """Query one EasyRAG workspace in a single async lifecycle."""

    rag = EasyRAG(working_dir=get_rag_working_dir(), workspace=get_rag_workspace())
    await rag.initialize_storages()
    try:
        result = await rag.aquery(
            query,
            QueryParam(
                mode="naive",
                chunk_top_k=k,
                top_k=k,
                rewrite_enabled=False,
                mqe_enabled=False,
            ),
        )
        return list(result.chunks)
    finally:
        await rag.finalize_storages()


def search_docs(query: str, k: int = 5) -> list[Document]:
    """
    Retrieve repository chunks through the EasyRAG naive mode compatibility path.

    Returns an empty list when the storage workspace has not been built yet.
    """

    try:
        return _run_async(_search(query, k))
    except Exception:
        return []
