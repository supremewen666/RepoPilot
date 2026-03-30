"""Build the local RAG index for RepoPilot documentation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repopilot.config import get_rag_index_path, get_repo_root
from repopilot.rag.indexer import build_vector_index, chunk_documents, load_repo_documents


def main() -> None:
    """
    Build or refresh the repo-local documentation index used by RepoPilot.

    Why:
        RAG retrieval is only useful after the repository's docs have been read,
        chunked, and serialized into the local index file. This script gives the
        project a single explicit command for that setup step.
    """

    repo_root = get_repo_root()
    documents = load_repo_documents(str(repo_root))
    chunks = chunk_documents(documents)
    build_vector_index(documents)

    print(f"repo_root={repo_root}")
    print(f"documents={len(documents)}")
    print(f"chunks={len(chunks)}")
    print(f"index_path={get_rag_index_path()}")


if __name__ == "__main__":
    main()
