## CODEX.md

This directory documents RepoPilot's teaching-first `RAG-from-scratch` layout.

## Canonical Architecture

The canonical code path now follows the same order we want learners to read:

1. `repopilot/rag/indexing/`
2. `repopilot/rag/knowledge/`
3. `repopilot/rag/storage/`
4. `repopilot/rag/retrieval/`
5. `repopilot/rag/orchestrator.py`
6. `repopilot/service/`

## Core Modules

- `repopilot/rag/orchestrator.py`: `EasyRAG`, the only canonical RAG entry point.
- `repopilot/rag/types.py`: query/result/config dataclasses and callable aliases.
- `repopilot/rag/storage/base.py`: storage abstractions.
- `repopilot/rag/indexing/`: document discovery, chunking, insert preparation, and ingestion pipeline.
- `repopilot/rag/knowledge/`: KG extraction, graph-to-vector sync, and manual curation payload helpers.
- `repopilot/rag/retrieval/`: query preprocessing, candidate generation, fusion, hydration, and execution.
- `repopilot/rag/storage/`: backend bundles and storage implementations grouped by responsibility.
- `repopilot/service/`: FastAPI, task management, agent runtime, response shaping, and GitHub integration.
- `repopilot/support/optional_deps.py`: lightweight fallbacks when optional packages are unavailable.
- `repopilot/config/`: runtime, model, storage, and integration configuration helpers.

## Public API

Canonical `repopilot.rag` exports:

- `EasyRAG`
- `QueryParam`
- `QueryResult`
- `KGExtractionConfig`
- storage abstract base classes

Compatibility shims remain available in the old module paths, but they are no longer the documented source of truth.

## Teaching Docs

Read the walkthroughs in this order:

- `docs/tutorials/indexing.md`
- `docs/tutorials/storage.md`
- `docs/tutorials/retrieval.md`
- `docs/tutorials/orchestrator.md`
- `docs/tutorials/service.md`
