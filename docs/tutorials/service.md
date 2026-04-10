# Service Walkthrough

The service layer wraps the RAG core without owning its internal algorithms.

- Canonical path: `repopilot/service/`
- `api/` contains the FastAPI app and request schemas.
- `tasks/` contains persistent background index task handling.
- `agent/` contains prompts, fallback behavior, tools, and runtime orchestration.
- `integrations/` contains GitHub MCP wiring.

Legacy shim paths:

- `repopilot/api/`
- `repopilot/agent/`
- `repopilot/integrations/`
- `repopilot/schemas.py`
- `repopilot/response_builder.py`
