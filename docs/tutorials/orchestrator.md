# Orchestrator Walkthrough

`EasyRAG` is the single public entry point for the RAG core.

- Canonical path: `repopilot/rag/orchestrator.py`
- Read order: initialization, insert/delete lifecycle, query execution, manual KG curation helpers, stats.
- The orchestrator depends on canonical indexing, retrieval, knowledge, and storage modules only.

Legacy shim path:

- `repopilot/rag/easyrag.py`
