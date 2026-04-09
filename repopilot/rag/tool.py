"""LangChain-compatible tools for RepoPilot's EasyRAG subsystem."""

from __future__ import annotations

import asyncio
import json
from typing import Callable

from repopilot.compat import tool
from repopilot.config import get_rag_working_dir, get_rag_workspace
from repopilot.rag.base import QueryParam
from repopilot.rag.easyrag import EasyRAG
from repopilot.rag.retriever import search_docs


def _run_async(awaitable: object) -> object:
    """Run an async EasyRAG call from synchronous tool wrappers."""

    try:
        return asyncio.run(awaitable)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(awaitable)
        finally:
            loop.close()


def _serialize_citations(citations: list[dict[str, str]]) -> str:
    """Serialize citations into a compact JSON payload for the agent."""

    return json.dumps(citations, ensure_ascii=False, indent=2)


def create_search_docs_tool(
    rag_getter: Callable[[], EasyRAG],
    *,
    default_mode: str = "hybrid",
    rewrite_enabled: bool = True,
    mqe_enabled: bool = True,
):
    """Create a tool bound to a lazy EasyRAG instance."""

    @tool(description="Search repository knowledge chunks and return grounded citations.")
    def search_repo_knowledge(query: str) -> str:
        rag = rag_getter()
        result = _run_async(
            rag.aquery(
                query,
                QueryParam(
                    mode=default_mode,
                    top_k=8,
                    chunk_top_k=5,
                    enable_rerank=default_mode == "mix",
                    rewrite_enabled=rewrite_enabled,
                    mqe_enabled=mqe_enabled,
                ),
            )
        )
        return _serialize_citations(result.citations)

    return search_repo_knowledge


@tool
def search_docs_tool(query: str) -> str:
    """Compatibility wrapper that runs EasyRAG naive retrieval and returns JSON citations."""

    documents = search_docs(query=query, k=5)
    serialized = [
        {
            "source_type": document.metadata.get("source_type", "doc"),
            "title": document.metadata.get("title", "Document"),
            "location": document.metadata.get("path", ""),
            "snippet": document.page_content[:400].strip(),
        }
        for document in documents
    ]
    return _serialize_citations(serialized)


def get_default_rag_tool():
    """Return a default tool bound to the standard workspace."""

    rag = EasyRAG(working_dir=get_rag_working_dir(), workspace=get_rag_workspace())
    _run_async(rag.initialize_storages())
    mode = "mix" if rag.can_rerank() else "hybrid"
    return create_search_docs_tool(lambda: rag, default_mode=mode, rewrite_enabled=True, mqe_enabled=True)
