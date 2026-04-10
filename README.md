# RepoPilot

RepoPilot is a single-repository engineering assistant built for interview-demo scope. It combines an EasyRAG-style repository knowledge subsystem with PDF loading, Qwen3 embedding and reranking hooks, query rewriting + MQE preprocessing, graph curation APIs, an optional FastAPI service surface, GitHub MCP read-only access, lightweight long-term memory, and a Streamlit chat UI.

## What It Does

- Answers questions about repository documentation with citations
- Queries GitHub repository context through read-only MCP tools
- Remembers lightweight user preferences and ongoing task context
- Returns a structured response with answer, sources, memory usage, and confidence

## Scope

This project is intentionally narrow.

- Single repository only
- Read-only GitHub workflows
- No multi-agent orchestration
- No code modification or PR creation
- No enterprise auth or permissions system

## Architecture

- `streamlit_app.py`: chat UI and session state
- `repopilot/agent/runner.py`: agent construction and orchestration
- `repopilot/rag/`: EasyRAG-style repository knowledge subsystem with orchestrator, storage backends, and query modes
- `repopilot/integrations/github_mcp.py`: GitHub MCP tool loading and config parsing
- `repopilot/memory/store.py`: mem0-first memory with local JSON fallback
- `repopilot/response_builder.py`: normalizes agent output into the UI schema

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -e .
```

For test and local development tools, install the dev extras:

```bash
pip install -e ".[dev]"
```

`requirements.txt` is kept as a compatibility export of the same runtime dependency set for environments that still prefer `pip install -r requirements.txt`.
`python-dotenv` is included in the project dependencies, so `.env` values are loaded automatically at runtime.

### 3. Configure environment variables

Copy the example file and fill in the values you need:

```bash
cp .env.example .env
```

Minimum setup for a docs-only demo:

- `OPENAI_API_KEY`

Optional setup for GitHub MCP:

- `GITHUB_PERSONAL_ACCESS_TOKEN`
- `REPOPILOT_GITHUB_MCP_COMMAND`
- `REPOPILOT_GITHUB_MCP_ENV`

Optional setup for OpenAI-compatible providers:

- `OPENAI_BASE_URL`
- `REPOPILOT_MODEL`
- `REPOPILOT_MODEL_NAME`
- `REPOPILOT_QUERY_MODEL_NAME`
- `REPOPILOT_EMBEDDING_MODEL_NAME`
- `REPOPILOT_RERANK_MODEL_NAME`
- `REPOPILOT_KG_MODEL_NAME`
- `REPOPILOT_QUERY_BASE_URL`
- `REPOPILOT_EMBEDDING_BASE_URL`
- `REPOPILOT_RERANK_BASE_URL`
- `REPOPILOT_KG_BASE_URL`
- `REPOPILOT_KG_ENTITY_TYPES`

Optional setup for the production RAG backend bundle:

- `REPOPILOT_RAG_STORAGE_BACKEND=postgres_qdrant`
- `REPOPILOT_POSTGRES_DSN`
- `REPOPILOT_QDRANT_URL`
- `REPOPILOT_QDRANT_API_KEY`
- `REPOPILOT_QDRANT_COLLECTION_PREFIX`

DashScope notes:

- `OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1` works for chat/query models.
- Text embeddings can still use the OpenAI-compatible embeddings API.
- `Qwen3-VL-Embedding-*` now auto-switches to DashScope's official multimodal embedding endpoint when the embedding base URL points at a DashScope host.
- `qwen3-rerank`, `gte-rerank-v2`, and `Qwen3-VL-Reranker-*` now auto-switch to DashScope's official rerank endpoints when the rerank base URL points at a DashScope host.
- PDF pages with embedded images are now exported under `.repopilot/media/...` during indexing and those image assets are included in VL embedding/rerank requests.

## Build the Document Index

Before asking documentation questions, build the local RAG workspace:

```bash
python scripts/build_index.py
```

This performs a full sync over repo-local docs such as `README`, `docs/`, `.txt`, and text-based PDF documents, then writes EasyRAG-style storage files under `.repopilot/rag_storage/<workspace>/`.

Targeted maintenance is also supported:

```bash
python scripts/build_index.py --action rebuild --doc-id doc::docs-architecture-md
python scripts/build_index.py --action delete --doc-id doc::docs-architecture-md
```

`update` is an alias of `rebuild`. Rebuild by `doc_id` replaces the selected documents in place, while full rebuild removes stale indexed docs that no longer exist in the repository.

## Run the App

```bash
streamlit run streamlit_app.py
```

Open the local Streamlit URL in your browser and ask questions such as:

- `What does RepoPilot do?`
- `How is documentation retrieval implemented?`
- `What tools are exposed to the agent?`

## Run the Service

RepoPilot also exposes the RAG/KG layer through FastAPI:

```bash
uvicorn repopilot.service.api.app:create_app --factory --reload
```

The first service cut includes:

- `POST /rag/query`
- `POST /rag/index/tasks`
- `GET /rag/index/tasks`
- `GET /rag/index/tasks/{task_id}`
- `POST /rag/kg/entities`
- `PATCH /rag/kg/entities/{entity_id}`
- `DELETE /rag/kg/entities/{entity_id}`
- `POST /rag/kg/entities/merge`
- `POST /rag/kg/relations`
- `PATCH /rag/kg/relations/{relation_id}`
- `DELETE /rag/kg/relations/{relation_id}`
- `POST /rag/kg/relations/merge`
- `POST /rag/kg/custom`
- `GET /healthz`

## GitHub MCP Setup

RepoPilot supports GitHub MCP in read-only mode.

### Local stdio mode

1. Place the `github-mcp-server` binary at `bin/github-mcp-server`
2. Set `GITHUB_PERSONAL_ACCESS_TOKEN`
3. Keep `REPOPILOT_GITHUB_MCP_COMMAND` pointed at `scripts/start_github_mcp.sh`

You can smoke-test the MCP wiring with:

```bash
python scripts/test_github_mcp.py
```

If MCP is unavailable, the app still runs in docs-only demo mode.

## Testing

Run the test suite with:

```bash
python -m pytest -q
```

## Notes for Interview Demo

What this project shows well:

- clear product scope
- practical LangChain agent orchestration
- an EasyRAG-style RAG subsystem adapted for one repository
- query rewriting + MQE retrieval preprocessing
- dense embedding retrieval with token fallback
- bounded use of RAG, MCP, and memory
- graceful fallbacks when optional integrations are unavailable
- basic unit test coverage around config, RAG, memory, and response normalization

Known limitations:

- GitHub MCP citation grounding is less complete than the docs RAG path
- PDF support is limited to text-extractable PDFs and does not include OCR
- semantic chunking falls back to sliding windows when embeddings are unavailable
- memory persistence uses a narrow heuristic and a simple JSON fallback

## EasyRAG-Style Knowledge Layer

RepoPilot now models RAG as a small EasyRAG-style subsystem:

- `repopilot/rag/easyrag.py`: main orchestrator with async lifecycle
- `repopilot/rag/base.py`: pluggable storage abstractions
- `repopilot/rag/providers.py`: OpenAI-compatible query, embedding, rerank, and KG extraction hooks
- `repopilot/rag/chunking.py`: structured, semantic, and sliding-window chunking strategies
- `repopilot/rag/kg_extraction.py`: LLM KG extraction normalization plus heuristic fallback
- `repopilot/rag/preprocess.py`: query rewriting and MQE preprocessing
- `repopilot/rag/kg/`: built-in JSON, dense embedding, token fallback, graph, and status backends
- `repopilot/rag/production.py`: optional PostgreSQL and Qdrant storage backends
- `repopilot/api/`: FastAPI service layer and index task orchestration
- `repopilot/rag/operate.py`: document loading, PDF parsing, insertion, and multi-mode query execution

Index lifecycle behavior:

- `EasyRAG.ainsert()` / `EasyRAG.ainsert_documents()`: true upsert by `doc_id`
- `EasyRAG.adelete_documents()`: remove one or more indexed docs and all derived graph/vector state
- `EasyRAG.acreate_entity()` / `aupdate_entity()` / `adelete_entity()` / `amerge_entities()`: manual entity curation
- `EasyRAG.acreate_relation()` / `aupdate_relation()` / `adelete_relation()` / `amerge_relations()`: manual semantic relation curation
- `EasyRAG.ainsert_custom_kg()`: batch insert manual entities and relations into the primary graph/vector path
- `rebuild_document_index()`: full-sync the workspace, including stale-doc cleanup

KG extraction behavior:

- chunk ingestion now uses LLM-backed structured extraction when a compatible model is configured
- default entity types are architecture-oriented: `component,module,file,service,tool,workflow,concept,config,dependency,interface`
- if KG model calls fail or are not configured, ingestion falls back to the previous heuristic extraction path
- semantic relations are stored as first-class relation records, so merge/delete/manual insert operations update both graph state and relation vectors

Supported query modes:

- `naive`: direct dense chunk retrieval
- `local`: entity-neighborhood retrieval with dense backfill
- `global`: summary and central-entity retrieval with dense backfill
- `hybrid`: merge local and global after rewrite + MQE preprocessing
- `mix`: merge hybrid and naive, then rerank with `qwen3-rerank` when available

Chunking strategy selection:

- Markdown / MDX / RST / README: structured chunking
- TXT and long plain text: semantic chunking
- PDF: semantic chunking by default, or structured chunking when headings are detectable
- Any strategy failure: sliding-window fallback with overlap metadata

## Example Flow

1. Install dependencies
2. Configure `.env`
3. Build the RAG index
4. Start Streamlit
5. Ask a docs question first
6. If GitHub MCP is configured, ask about an issue, PR, or file
