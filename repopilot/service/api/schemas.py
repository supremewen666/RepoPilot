"""Pydantic request models for the RepoPilot RAG API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request payload for one RAG query."""

    query: str
    mode: str = "hybrid"
    workspace: str | None = None
    top_k: int = 8
    chunk_top_k: int = 6
    enable_rerank: bool = False


class IndexTaskRequest(BaseModel):
    """Request payload for one index maintenance task."""

    action: Literal["full_sync", "rebuild_docs", "delete_docs"]
    workspace: str | None = None
    doc_ids: list[str] = Field(default_factory=list)


class EntityPayload(BaseModel):
    """Mutable entity fields used by CRUD and custom KG APIs."""

    id: str | None = None
    label: str
    entity_types: list[str] = Field(default_factory=list)
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: list[str] = Field(default_factory=list)


class EntityUpdatePayload(BaseModel):
    """Patch payload for one entity."""

    label: str | None = None
    entity_types: list[str] | None = None
    description: str | None = None
    aliases: list[str] | None = None
    metadata: dict[str, Any] | None = None
    provenance: list[str] | None = None


class EntityMergeRequest(BaseModel):
    """Merge payload for entity deduplication."""

    source_entity_id: str
    target_entity_id: str


class RelationPayload(BaseModel):
    """Mutable relation fields used by CRUD and custom KG APIs."""

    id: str | None = None
    source_entity_id: str
    target_entity_id: str
    relation: str
    description: str = ""
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: list[str] = Field(default_factory=list)


class RelationUpdatePayload(BaseModel):
    """Patch payload for one semantic relation."""

    source_entity_id: str | None = None
    target_entity_id: str | None = None
    relation: str | None = None
    description: str | None = None
    weight: float | None = None
    metadata: dict[str, Any] | None = None
    provenance: list[str] | None = None


class RelationMergeRequest(BaseModel):
    """Merge payload for relation deduplication."""

    source_relation_id: str
    target_relation_id: str


class CustomKGRequest(BaseModel):
    """Manual KG insertion payload."""

    entities: list[EntityPayload] = Field(default_factory=list)
    relations: list[RelationPayload] = Field(default_factory=list)
    source_label: str | None = None
    batch_id: str | None = None


class QueryResponseChunk(BaseModel):
    """Serializable chunk returned by the query API."""

    page_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Serializable query result envelope."""

    mode: str
    chunks: list[QueryResponseChunk] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
