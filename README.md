# RepoPilot

RepoPilot is a single-repository engineering assistant built for interview-demo scope. It combines documentation RAG, GitHub MCP read-only access, lightweight long-term memory, and a Streamlit chat UI.

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
- `repopilot/rag/`: local document indexing and retrieval
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

## Build the Document Index

Before asking documentation questions, build the local RAG index:

```bash
python scripts/build_index.py
```

This scans repo-local docs such as `README`, `docs/`, and other text-based documentation, then writes a lightweight JSON index under `.repopilot/`.

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
- bounded use of RAG, MCP, and memory
- graceful fallbacks when optional integrations are unavailable
- basic unit test coverage around config, RAG, memory, and response normalization

Known limitations:

- GitHub MCP citation grounding is less complete than the docs RAG path
- RAG retrieval is lightweight token-overlap retrieval, not embeddings
- memory persistence uses a narrow heuristic and a simple JSON fallback

## Example Flow

1. Install dependencies
2. Configure `.env`
3. Build the RAG index
4. Start Streamlit
5. Ask a docs question first
6. If GitHub MCP is configured, ask about an issue, PR, or file
