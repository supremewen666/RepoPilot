"""Tests for lightweight RAG indexing and retrieval."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from repopilot.rag.indexer import build_vector_index, load_repo_documents
from repopilot.rag.retriever import search_docs


class RAGRetrieverTestCase(unittest.TestCase):
    """Verify the offline index and online retrieval path."""

    def test_rag_index_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "architecture.md").write_text(
                "# Architecture\nRepoPilot uses LangChain for workflow orchestration.\n",
                encoding="utf-8",
            )

            old_value = os.environ.get("REPOPILOT_RAG_INDEX_PATH")
            os.environ["REPOPILOT_RAG_INDEX_PATH"] = str(root / "rag_index.json")
            try:
                documents = load_repo_documents(str(root))
                build_vector_index(documents)
                results = search_docs("How does workflow orchestration work?", k=3)
            finally:
                if old_value is None:
                    os.environ.pop("REPOPILOT_RAG_INDEX_PATH", None)
                else:
                    os.environ["REPOPILOT_RAG_INDEX_PATH"] = old_value

        self.assertTrue(results)
        self.assertIn("LangChain", results[0].page_content)


if __name__ == "__main__":
    unittest.main()
