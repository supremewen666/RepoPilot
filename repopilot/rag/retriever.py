"""Online retrieval helpers for RepoPilot's document RAG layer."""

from __future__ import annotations

import json
import re

from repopilot.compat import Document
from repopilot.config import get_rag_index_path


def _tokenize(text: str) -> set[str]:
    """Normalize a query or chunk into a set of lowercase retrieval terms."""

    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def search_docs(query: str, k: int = 5) -> list[Document]:
    """
    Retrieve the most relevant documentation chunks for a user query.

    Inputs:
        - query: natural-language question from the user
        - k: maximum number of ranked chunks to return

    Returns:
        Ranked chunks only. This function deliberately does not synthesize the
        final answer so retrieval stays deterministic and easy to unit test.

    Failure strategy:
        If the local index does not exist yet, return an empty list rather than
        throwing. The caller can then explain that documentation indexing has
        not been prepared.
    """

    index_path = get_rag_index_path()
    if not index_path.exists():
        return []

    records = json.loads(index_path.read_text(encoding="utf-8"))
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored: list[tuple[int, dict]] = []
    for record in records:
        tokens = set(record.get("tokens", []))
        score = len(query_tokens & tokens)
        if score:
            scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    results: list[Document] = []
    for _, record in scored[:k]:
        results.append(
            Document(
                page_content=record["page_content"],
                metadata=record["metadata"],
            )
        )
    return results
