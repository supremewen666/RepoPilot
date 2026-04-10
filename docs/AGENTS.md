# Repository Guidelines

This repository now uses a teaching-first RAG layout.

## Canonical Module Order

- `repopilot/rag/indexing/`: discovery, PDF loading, input preparation, chunking, ingestion.
- `repopilot/rag/knowledge/`: KG extraction and graph/vector synchronization.
- `repopilot/rag/storage/`: storage abstractions and backends grouped by responsibility.
- `repopilot/rag/retrieval/`: preprocessing, retrieval modes, fusion, hydration, execution.
- `repopilot/rag/orchestrator.py`: `EasyRAG`.
- `repopilot/service/`: API, tasks, agent runtime, integrations, response models.

## Compatibility Rule

- Old paths under `repopilot/rag/operate.py`, `repopilot/rag/indexer.py`, `repopilot/rag/retriever.py`, `repopilot/rag/tool.py`, `repopilot/rag/kg/`, `repopilot/api/`, and `repopilot/agent/` are compatibility shims.
- Do not add new core behavior to shim modules.
- New logic belongs in canonical modules only.

## Development Commands

- `python scripts/build_index.py`
- `python -m pytest -q`
- `ruff check .`

## Design Constraints

- RAG core modules must not import `repopilot.service`.
- Service entrypoints should depend on `repopilot.rag` public APIs, not on canonical indexing/retrieval internals.
- Keep comments and user-visible backend text in English.
- Prefer explicit single-purpose modules over convenience facades.
