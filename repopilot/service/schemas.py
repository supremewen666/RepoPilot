"""Canonical response and source schemas for RepoPilot."""

from __future__ import annotations

from repopilot.support.optional_deps import BaseModel, Field


class Citation(BaseModel):
    """
    One evidence item used in the final answer.

    Fields are intentionally UI-oriented so Streamlit can render citations
    without understanding whether the source came from RAG or GitHub MCP.
    """

    source_type: str = Field(default="")
    label: str = Field(default="")
    url_or_path: str = Field(default="")
    snippet: str = Field(default="")


class SourceItem(BaseModel):
    """
    Shared internal source shape used before converting into final citations.

    This keeps tool-specific adapters simple: each integration can normalize
    its raw output once, then the response builder handles the rest.
    """

    source_type: str = Field(default="")
    title: str = Field(default="")
    location: str = Field(default="")
    snippet: str = Field(default="")


class FinalResponse(BaseModel):
    """
    Canonical response contract between backend orchestration and Streamlit UI.

    Every backend path should return this structure so the UI never needs to
    parse model text or integration-specific payloads to show sources.
    """

    answer: str = Field(default="")
    citations: list[Citation] = Field(default_factory=list)
    used_memory: list[str] = Field(default_factory=list)
    confidence: str = Field(default="low")
