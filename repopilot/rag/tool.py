"""LangChain tool wrapper for document retrieval."""

from __future__ import annotations

import json

from repopilot.compat import tool
from repopilot.rag.retriever import search_docs


@tool
def search_docs_tool(query: str) -> str:
    """
    LangChain tool wrapper over the retrieval layer.

    Model-visible capability:
        Let the agent ask for documentation evidence relevant to a question.

    Explicit non-goals:
        This tool does not invent answers, modify repository state, or inspect
        GitHub metadata. It only returns retrieved documentation chunks.

    Output:
        A compact JSON string containing snippet text and citation metadata so
        downstream answer assembly can convert results into FinalResponse items.
    """

    documents = search_docs(query=query, k=5)
    serialized = [
        {
            "source_type": document.metadata.get("source_type", "doc"),
            "title": document.metadata.get("title", "Document"),
            "location": document.metadata.get("path", ""),
            "snippet": document.page_content[:400],
        }
        for document in documents
    ]
    return json.dumps(serialized, ensure_ascii=False, indent=2)
