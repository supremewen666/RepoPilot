"""Build the local RAG index for RepoPilot documentation."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repopilot.config import get_rag_working_dir, get_rag_workspace, get_repo_root  # noqa: E402
from repopilot.rag import EasyRAG, load_repo_documents  # noqa: E402


def _run_async(awaitable: object) -> object:
    """Run async EasyRAG operations from the build script."""

    try:
        return asyncio.run(awaitable)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(awaitable)
        finally:
            loop.close()


def main() -> None:
    """
    Build or refresh the repo-local documentation index used by RepoPilot.

    Why:
        RAG retrieval is only useful after the repository's docs have been read,
        chunked, and serialized into the local index file. This script gives the
        project a single explicit command for that setup step.
    """

    repo_root = get_repo_root()
    documents = load_repo_documents(repo_root)
    rag = EasyRAG(working_dir=get_rag_working_dir(), workspace=get_rag_workspace())
    _run_async(rag.initialize_storages())
    try:
        stats = _run_async(rag.ainsert_documents(documents))
        aggregate = _run_async(rag.get_stats())
    finally:
        _run_async(rag.finalize_storages())

    print(f"repo_root={repo_root}")
    print(f"workspace={rag.workspace}")
    print(f"working_dir={rag.workspace_dir}")
    print(f"documents={stats.get('documents', 0)}")
    print(f"pdf_documents={stats.get('pdf_documents', 0)}")
    print(f"chunks={stats.get('chunks', 0)}")
    print(f"entities={aggregate.get('entity_vectors', 0)}")
    print(f"relations={aggregate.get('relation_vectors', 0)}")
    print(f"chunk_strategy_counts={aggregate.get('chunk_strategy_counts', {})}")
    print(f"vector_backend={aggregate.get('vector_backend', 'unknown')}")


if __name__ == "__main__":
    main()
