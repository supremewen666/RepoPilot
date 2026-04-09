# RepoPilot

RepoPilot is a single-repository engineering assistant built for interview-demo scope. It combines an EasyRAG-style repository knowledge subsystem with PDF loading, Qwen3 embedding and reranking hooks, query rewriting + MQE preprocessing, GitHub MCP read-only access, lightweight long-term memory, and a Streamlit chat UI.

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
pip install -r requirements.txt
```

`python-dotenv` is included in `requirements.txt`, so `.env` values are loaded automatically at runtime.

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
- `REPOPILOT_QUERY_BASE_URL`
- `REPOPILOT_EMBEDDING_BASE_URL`
- `REPOPILOT_RERANK_BASE_URL`

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

This scans repo-local docs such as `README`, `docs/`, `.txt`, and text-based PDF documents, then writes EasyRAG-style storage files under `.repopilot/rag_storage/<workspace>/`.

## Run the App

```bash
streamlit run streamlit_app.py
```

Open the local Streamlit URL in your browser and ask questions such as:

- `What does RepoPilot do?`
- `How is documentation retrieval implemented?`
- `What tools are exposed to the agent?`

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
- graph/entity extraction uses deterministic heuristics rather than an LLM extraction pipeline
- PDF support is limited to text-extractable PDFs and does not include OCR
- semantic chunking falls back to sliding windows when embeddings are unavailable
- memory persistence uses a narrow heuristic and a simple JSON fallback

## EasyRAG-Style Knowledge Layer

RepoPilot now models RAG as a small EasyRAG-style subsystem:

- `repopilot/rag/easyrag.py`: main orchestrator with async lifecycle
- `repopilot/rag/base.py`: pluggable storage abstractions
- `repopilot/rag/providers.py`: OpenAI-compatible Qwen3 query, embedding, and rerank hooks
- `repopilot/rag/chunking.py`: structured, semantic, and sliding-window chunking strategies
- `repopilot/rag/preprocess.py`: query rewriting and MQE preprocessing
- `repopilot/rag/kg/`: built-in JSON, dense embedding, token fallback, graph, and status backends
- `repopilot/rag/operate.py`: document loading, PDF parsing, insertion, and multi-mode query execution

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
