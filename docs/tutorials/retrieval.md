# Retrieval Walkthrough

Read this layer for the query path.

- Canonical path: `repopilot/rag/retrieval/`
- `preprocess.py` handles normalization, rewrite, and MQE.
- `query_modes.py` generates naive, local, global, and relation candidates.
- `fusion.py` handles weighted merge and RRF.
- `hydration.py` turns ranked ids into final chunks and citations.
- `pipeline.py` is the end-to-end query orchestrator.

Legacy shim paths:

- `repopilot/rag/preprocess.py`
- `repopilot/rag/retrieve.py`
- `repopilot/rag/retriever.py`
