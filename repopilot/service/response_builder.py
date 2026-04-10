"""Helpers for normalizing raw agent output into the app's response schema."""

from __future__ import annotations

from typing import Any

from repopilot.service.schemas import Citation, FinalResponse, SourceItem


def _coerce_citation(item: Any) -> Citation:
    """Convert supported source-like objects into the public citation schema."""

    if isinstance(item, Citation):
        return item
    if isinstance(item, SourceItem):
        return Citation(
            source_type=item.source_type,
            label=item.title,
            url_or_path=item.location,
            snippet=item.snippet,
        )
    if isinstance(item, dict):
        if {"source_type", "title", "location", "snippet"}.issubset(item.keys()):
            source_item = SourceItem.model_validate(item)
            return Citation(
                source_type=source_item.source_type,
                label=source_item.title,
                url_or_path=source_item.location,
                snippet=source_item.snippet,
            )
        return Citation.model_validate(item)
    raise TypeError(f"Unsupported citation payload: {type(item)!r}")


def build_final_response(agent_output: dict[str, Any]) -> FinalResponse:
    """
    Normalize raw agent output into the single response schema used by the app.

    Why:
        Different execution paths may return different raw shapes: the LangChain
        branch can produce structured dictionaries, while fallback code may
        construct simpler payloads. This function keeps the UI insulated from
        those differences and enforces a single answer contract.
    """

    raw_citations = agent_output.get("citations", [])
    citations = [_coerce_citation(item) for item in raw_citations]
    answer = agent_output.get("answer", "").strip()
    if not answer:
        answer = "I could not produce a grounded answer from the available sources."

    return FinalResponse(
        answer=answer,
        citations=citations,
        used_memory=list(agent_output.get("used_memory", [])),
        confidence=agent_output.get("confidence", "low"),
    )
