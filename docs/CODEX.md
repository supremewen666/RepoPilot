## CODEX.md

This directory contains RepoPilot's EasyRAG-style repository knowledge subsystem.

## Project Overview

RepoPilot embeds a single-repository RAG framework that borrows the structure of EasyRAG while staying lightweight enough for an interview-demo scope. The subsystem indexes repository text and PDF documents, extracts heuristic entities and relations, persists graph and dense vector state locally, and exposes multi-mode retrieval with query rewriting and MQE preprocessing to the agent layer.

## Core Architecture

### Key Components

- `easyrag.py`: main orchestrator class (`EasyRAG`). Always call `await rag.initialize_storages()` before `ainsert()` or `aquery()`, and `await rag.finalize_storages()` when the process is done persisting state.
- `operate.py`: document loading, PDF parsing, chunk strategy selection, entity and relation extraction, insertion pipeline, and query-mode execution.
- `base.py`: abstract base classes for `BaseKVStorage`, `BaseVectorStorage`, `BaseGraphStorage`, and `BaseDocStatusStorage`, plus `QueryParam`.
- `chunking.py`: `structured`, `semantic`, and `sliding_window` chunkers with overlap-aware metadata.
- `preprocess.py`: query normalization, rewriting, and MQE preparation.
- `providers.py`: default OpenAI-compatible query, embedding, and rerank providers. The intended defaults are `qwen3-embedding` and `qwen3-rerank`.
- `kg/`: built-in single-node backends. Current defaults are `JSONKVStorage`, `EmbeddingVectorStorage`, `NetworkXGraphStorage`, and `JSONDocStatusStorage`.
- `indexer.py`, `retriever.py`, `tool.py`: compatibility layers that map the older RepoPilot API onto the new `EasyRAG` core.

### Storage Layer

RepoPilot uses four pluggable storage categories:

- `KV_STORAGE`: documents, chunks, summaries, and optional cache entries
- `VECTOR_STORAGE`: dense chunk, entity, relation, and summary retrieval records with token fallback
- `GRAPH_STORAGE`: entity/chunk/document graph structure
- `DOC_STATUS_STORAGE`: per-document indexing status

Workspace isolation uses file-based subdirectories under `.repopilot/rag_storage/<workspace>/`.

### Query Modes

- `local`: retrieve graph-neighbor chunks around query entities with dense backfill
- `global`: retrieve broad summary and central-entity context with dense backfill
- `hybrid`: combine local and global results after query rewriting and MQE
- `naive`: direct chunk search from dense vector storage
- `mix`: combine hybrid and naive, then rerank when a reranker is configured

## Development Notes

- Keep comments and user-visible backend text in English.
- Prefer deterministic behavior over hidden magic. If a heuristic is used, make it explicit and easy to test.
- The graph and vector layers must stay usable without external services by falling back gracefully when model calls fail.
- Do not mix GitHub MCP repository metadata into this package's persisted knowledge stores. This package is for repository docs and derived knowledge only.
- When extending the subsystem, prefer adding new storage implementations under `kg/` or new orchestration behavior in `operate.py` instead of bloating the compatibility wrappers.

## Critical Usage Pattern

```python
import asyncio

from repopilot.rag import EasyRAG, QueryParam


async def main() -> None:
    rag = EasyRAG()
    await rag.initialize_storages()
    await rag.ainsert(["Architecture notes"], ids=["doc::architecture"])
    result = await rag.aquery("How is indexing designed?", QueryParam(mode="hybrid"))
    print(result.citations)
    await rag.finalize_storages()


asyncio.run(main())
```
