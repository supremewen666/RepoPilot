# Indexing Walkthrough

Start here for the write path.

- Canonical path: `repopilot/rag/indexing/`
- Main flow: `loaders.py -> prepare.py -> chunking.py -> pipeline.py`
- Read `loaders.py` for repo doc discovery and PDF page handling.
- Read `chunking.py` for strategy choice and chunk orchestration.
- Read `pipeline.py` for the payloads that feed KV, vector, graph, and status storage.

Legacy shim paths:

- `repopilot/rag/documents.py`
- `repopilot/rag/ingest.py`
- `repopilot/rag/operate.py`
