# Storage Walkthrough

This layer explains where indexed state lives.

- Canonical path: `repopilot/rag/storage/`
- Abstract contracts live in `storage/base.py`.
- Backend bundle selection lives in `storage/bundles.py`.
- Local single-node implementations are exposed by `kv/`, `vector/`, `graph/`, and `status/`.
- Production PostgreSQL/Qdrant implementations are exposed through the same subpackages.

Legacy shim paths:

- `repopilot/rag/backends.py`
- `repopilot/rag/kg/`
- `repopilot/rag/production.py`
