"""Offline indexing utilities for RepoPilot documentation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from repopilot.compat import Document
from repopilot.config import get_rag_index_path

_TEXT_SUFFIXES = {".md", ".mdx", ".rst", ".txt"}
_EXCLUDED_PARTS = {".git", ".venv", "__pycache__", "node_modules", ".repopilot"}


def load_repo_documents(repo_root: str) -> list[Document]:
    """
    Load documentation files eligible for indexing.

    Include:
        README files, docs folders, design notes, ADRs, and plain text setup docs

    Exclude:
        source code, generated artifacts, virtual environment directories, and
        large binary or build-output folders that would pollute documentation RAG

    Why:
        This assistant treats RAG as a documentation layer. Code and PR details
        belong to GitHub MCP rather than being mixed into the doc index.
    """

    root = Path(repo_root).resolve()
    documents: list[Document] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _EXCLUDED_PARTS for part in path.parts):
            continue
        if path.suffix.lower() not in _TEXT_SUFFIXES and not path.name.lower().startswith("readme"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source_type": "doc",
                    "path": str(path),
                    "title": path.stem,
                },
            )
        )
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into retrieval-sized chunks while preserving source metadata.

    Important:
        Each chunk carries the original document path, title, and a chunk id so
        later answer assembly can cite the exact source instead of a vague file.

    Boundary handling:
        This uses a simple character window with overlap to keep the project
        lightweight. It favors deterministic behavior over embedding-specific
        splitting logic in the initial implementation.
    """

    chunk_size = 900
    overlap = 150
    chunks: list[Document] = []
    for document in documents:
        text = document.page_content
        start = 0
        chunk_id = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end].strip()
            if chunk_text:
                metadata = dict(document.metadata)
                metadata["chunk_id"] = chunk_id
                chunks.append(Document(page_content=chunk_text, metadata=metadata))
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
            chunk_id += 1
    return chunks


def _tokenize(text: str) -> list[str]:
    """Turn a text chunk into lowercase word tokens for lightweight retrieval."""

    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def build_vector_index(documents: list[Document]) -> None:
    """
    Create or refresh the local vector store used for documentation retrieval.

    Why:
        The project needs an offline, repo-local index so Streamlit can answer
        questions without rebuilding retrieval state on every request.

    Current implementation:
        Store chunk text, metadata, and pre-tokenized terms in JSON. This keeps
        the interface stable while leaving room to swap in a vector database
        later without changing the rest of the application.
    """

    chunks = chunk_documents(documents)
    payload = [
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
            "tokens": _tokenize(chunk.page_content),
        }
        for chunk in chunks
    ]
    index_path = get_rag_index_path()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
