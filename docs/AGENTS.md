# Repository Guidelines

This directory contains RepoPilot's EasyRAG-style repository knowledge subsystem.

## Project Structure & Module Organization
- `easyrag.py`: orchestrator and lifecycle entry point.
- `operate.py`: document loading, PDF parsing, document ingestion, heuristic entity and relation extraction, and multi-mode query logic.
- `base.py`: abstract storage interfaces and shared query/result types.
- `chunking.py`: strategy-specific chunking implementations.
- `preprocess.py`: query normalization, rewriting, and MQE expansion logic.
- `providers.py`: OpenAI-compatible query, embedding, and rerank providers.
- `kg/`: built-in local storage implementations.
- `indexer.py`, `retriever.py`, `tool.py`: compatibility layers for the older RepoPilot RAG API.

## Build, Test, and Development Commands
- `python scripts/build_index.py`: scan repo documentation and populate the default EasyRAG workspace.
- `python -m pytest -q`: run the project test suite.
- `ruff check .`: lint Python sources before committing.

## Coding Style & Naming Conventions
- Follow PEP 8 with four-space indentation and type annotations.
- Prefer dataclasses for state and small transport objects.
- Keep comments, docstrings, and backend messages in English.
- Use deterministic heuristics unless a configurable callable is explicitly injected.
- Treat PDF parsing, semantic chunking, dense retrieval, and query rewriting as separate responsibilities; do not collapse them into one helper.

## Testing Guidelines
- Add tests near the touched behavior under `tests/`.
- Cover lifecycle, persistence, PDF loading, chunk-strategy selection, query preprocessing, and multi-mode retrieval behavior.
- When compatibility wrappers change, keep smoke coverage for `search_docs()` and `search_docs_tool()`.

## Security & Configuration Tips
- Persist knowledge state only under `.repopilot/rag_storage/<workspace>/`.
- Do not store secrets, GitHub write data, or user memory state inside this package's storage files.
- External vector databases or graph services are intentionally out of scope for the default implementation.
- PDF support only targets text-extractable PDFs; OCR and scanned-image workflows are out of scope here.
