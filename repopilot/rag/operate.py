"""Compatibility facade for EasyRAG document and retrieval operations."""

from __future__ import annotations

from repopilot.rag.chunking import ChunkingConfig
from repopilot.rag.documents import PdfReader, chunk_documents, load_repo_documents, prepare_documents_for_insert
from repopilot.rag.ingest import ingest_documents
from repopilot.rag.retrieve import execute_query

__all__ = [
    "ChunkingConfig",
    "PdfReader",
    "chunk_documents",
    "execute_query",
    "ingest_documents",
    "load_repo_documents",
    "prepare_documents_for_insert",
]
