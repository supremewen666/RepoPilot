"""Tests for the EasyRAG-style repository knowledge subsystem."""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from repopilot.compat import Document
from repopilot.rag import EasyRAG, QueryParam, build_vector_index, search_docs, search_docs_tool
from repopilot.rag.chunking import ChunkingConfig
from repopilot.rag.indexer import load_repo_documents
from repopilot.rag.operate import chunk_documents
from scripts import build_index

_KEYWORDS = [
    "architecture",
    "setup",
    "langchain",
    "easyrag",
    "streamlit",
    "github",
    "query",
    "rewrite",
    "pdf",
    "semantic",
]


def _run(awaitable: object) -> object:
    """Run an async helper inside unittest."""

    return asyncio.run(awaitable)


def _stub_embedding(texts: list[str]) -> list[list[float]]:
    """Return deterministic dense embeddings for tests."""

    vectors: list[list[float]] = []
    for text in texts:
        lowered = text.lower()
        vector = [float(lowered.count(keyword)) for keyword in _KEYWORDS]
        vector.append(float(len(lowered.split())))
        vectors.append(vector)
    return vectors


def _stub_query_model(prompt: str, *, task: str, count: int = 1) -> str | list[str]:
    """Return deterministic query rewrites and MQE variants."""

    cleaned = prompt.split(":", 1)[-1].strip()
    if task == "rewrite":
        return f"{cleaned} repository architecture"
    if task == "mqe":
        return [f"{cleaned} variant {index}" for index in range(1, count + 1)]
    raise ValueError(task)


def _stub_reranker(query: str, items: list[dict[str, object]]) -> list[dict[str, object]]:
    """Rerank candidates by keyword overlap with the query."""

    lowered_query = query.lower()
    scored = []
    for item in items:
        text = str(item.get("text", "")).lower()
        score = sum(text.count(keyword) for keyword in lowered_query.split())
        candidate = dict(item)
        candidate["rerank_score"] = score
        scored.append(candidate)
    scored.sort(key=lambda item: float(item.get("rerank_score", 0.0)), reverse=True)
    return scored


class EasyRAGLifecycleTestCase(unittest.TestCase):
    """Verify lifecycle, persistence, and query modes."""

    def test_requires_initialize_before_use(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rag = EasyRAG(working_dir=tmp_dir, workspace="lifecycle", embedding_func=_stub_embedding, query_model_func=_stub_query_model)
            with self.assertRaises(RuntimeError):
                _run(rag.aquery("architecture", QueryParam(mode="naive")))
            with self.assertRaises(RuntimeError):
                _run(rag.ainsert("Architecture notes"))

    def test_insert_and_query_modes_with_preprocessing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rag = EasyRAG(
                working_dir=tmp_dir,
                workspace="repo",
                embedding_func=_stub_embedding,
                query_model_func=_stub_query_model,
                reranker_func=_stub_reranker,
            )
            _run(rag.initialize_storages())
            try:
                stats = _run(
                    rag.ainsert(
                        [
                            "# Architecture\nRepoPilot uses LangChain and EasyRAG for repository guidance.\n## Retrieval\nHybrid retrieval uses semantic chunks.\n",
                            "# Setup\nUse Streamlit and GitHub MCP to answer docs and PR questions.\n",
                        ],
                        ids=["doc::architecture", "doc::setup"],
                        file_paths=["docs/architecture.md", "docs/setup.md"],
                    )
                )
                self.assertEqual(stats["documents"], 2)

                for mode in ("naive", "local", "global", "hybrid", "mix"):
                    result = _run(
                        rag.aquery(
                            "How does LangChain help repository guidance?",
                            QueryParam(mode=mode, chunk_top_k=3, enable_rerank=(mode in {"hybrid", "mix"})),
                        )
                    )
                    self.assertTrue(result.citations, mode)
                    self.assertEqual(result.mode, mode)
                    self.assertIn("rewritten_query", result.metadata)
                    self.assertIn("expanded_queries", result.metadata)
                    self.assertIn("retrieval_queries", result.metadata)

                aggregate = _run(rag.get_stats())
                self.assertGreaterEqual(aggregate["graph_nodes"], 4)
                self.assertGreaterEqual(aggregate["entity_vectors"], 1)
                self.assertEqual(aggregate["vector_backend"], "dense_embedding")
            finally:
                _run(rag.finalize_storages())

    def test_workspace_isolation_and_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rag_alpha = EasyRAG(working_dir=tmp_dir, workspace="alpha", embedding_func=_stub_embedding, query_model_func=_stub_query_model)
            rag_beta = EasyRAG(working_dir=tmp_dir, workspace="beta", embedding_func=_stub_embedding, query_model_func=_stub_query_model)

            _run(rag_alpha.initialize_storages())
            _run(rag_beta.initialize_storages())
            try:
                _run(rag_alpha.ainsert("Alpha design uses Streamlit.", ids=["doc::alpha"], file_paths=["docs/alpha.md"]))
                _run(rag_beta.ainsert("Beta design uses EasyRAG.", ids=["doc::beta"], file_paths=["docs/beta.md"]))
            finally:
                _run(rag_alpha.finalize_storages())
                _run(rag_beta.finalize_storages())

            reopened_alpha = EasyRAG(working_dir=tmp_dir, workspace="alpha", embedding_func=_stub_embedding, query_model_func=_stub_query_model)
            reopened_beta = EasyRAG(working_dir=tmp_dir, workspace="beta", embedding_func=_stub_embedding, query_model_func=_stub_query_model)
            _run(reopened_alpha.initialize_storages())
            _run(reopened_beta.initialize_storages())
            try:
                alpha_result = _run(reopened_alpha.aquery("What uses Streamlit?", QueryParam(mode="naive", rewrite_enabled=False, mqe_enabled=False)))
                beta_result = _run(reopened_beta.aquery("What uses EasyRAG?", QueryParam(mode="naive", rewrite_enabled=False, mqe_enabled=False)))
                self.assertTrue(alpha_result.citations)
                self.assertTrue(beta_result.citations)
                self.assertIn("Streamlit", alpha_result.citations[0]["snippet"])
                self.assertIn("EasyRAG", beta_result.citations[0]["snippet"])
            finally:
                _run(reopened_alpha.finalize_storages())
                _run(reopened_beta.finalize_storages())

            self.assertTrue((Path(tmp_dir) / "alpha" / "kv" / "documents.json").exists())
            self.assertTrue((Path(tmp_dir) / "beta" / "vector" / "chunk.npy").exists())

    def test_dense_failure_falls_back_to_token_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            def failing_embedding(_: list[str]) -> list[list[float]]:
                raise RuntimeError("embedding failure")

            rag = EasyRAG(working_dir=tmp_dir, workspace="fallback", embedding_func=failing_embedding, query_model_func=_stub_query_model)
            _run(rag.initialize_storages())
            try:
                _run(
                    rag.ainsert(
                        ["# Architecture\nRepoPilot uses LangChain for workflow orchestration.\n"],
                        ids=["doc::architecture"],
                        file_paths=["docs/architecture.md"],
                    )
                )
                result = _run(rag.aquery("workflow orchestration", QueryParam(mode="naive", rewrite_enabled=False, mqe_enabled=False)))
            finally:
                _run(rag.finalize_storages())

            self.assertTrue(result.citations)
            self.assertEqual(result.metadata["vector_backend"], "fallback_token")


class ChunkingAndLoadingTestCase(unittest.TestCase):
    """Verify loading and chunk strategy selection."""

    def test_chunk_documents_chooses_structured_and_semantic_strategies(self) -> None:
        documents = [
            Document(
                page_content="# Architecture\nIntro.\n## Retrieval\nSemantic chunks help retrieval.\n",
                metadata={"doc_id": "doc::md", "path": "docs/architecture.md", "relative_path": "docs/architecture.md", "title": "architecture", "source_type": "doc"},
            ),
            Document(
                page_content="Sentence one about semantic retrieval. Sentence two about Qwen. Sentence three about overlap.",
                metadata={"doc_id": "doc::txt", "path": "docs/notes.txt", "relative_path": "docs/notes.txt", "title": "notes", "source_type": "doc"},
            ),
        ]
        rag = EasyRAG(working_dir="/tmp", workspace="unused", embedding_func=_stub_embedding, query_model_func=_stub_query_model)
        chunks = chunk_documents(documents, config=ChunkingConfig(), rag=rag)
        strategies = {str(chunk.metadata.get("chunk_strategy")) for chunk in chunks}

        self.assertIn("structured", strategies)
        self.assertIn("semantic", strategies)
        self.assertTrue(all(chunk.metadata.get("overlap_policy") for chunk in chunks))

    def test_load_repo_documents_includes_pdf_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "design.md").write_text("# Design\nGraph retrieval.\n", encoding="utf-8")
            pdf_path = docs_dir / "manual.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 fake")

            fake_pages = [
                mock.Mock(extract_text=mock.Mock(return_value="Page one architecture notes")),
                mock.Mock(extract_text=mock.Mock(return_value="")),
                mock.Mock(extract_text=mock.Mock(return_value="Page three setup details")),
            ]
            fake_reader = mock.Mock(pages=fake_pages)
            with mock.patch("repopilot.rag.documents.PdfReader", return_value=fake_reader):
                documents = load_repo_documents(root)

            pdf_documents = [document for document in documents if document.metadata["source_type"] == "pdf"]
            self.assertEqual(len(pdf_documents), 2)
            self.assertEqual(pdf_documents[0].metadata["page_number"], 1)
            self.assertEqual(pdf_documents[1].metadata["page_number"], 3)

    def test_load_repo_documents_keeps_image_only_pdf_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            pdf_path = docs_dir / "visual-manual.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 fake")

            fake_image = mock.Mock(name="diagram.png", data=b"\x89PNG\r\n\x1a\n")
            fake_pages = [
                mock.Mock(extract_text=mock.Mock(return_value=""), images=[fake_image]),
            ]
            fake_reader = mock.Mock(pages=fake_pages)
            with mock.patch("repopilot.rag.documents.PdfReader", return_value=fake_reader):
                documents = load_repo_documents(root)

            self.assertEqual(len(documents), 1)
            self.assertIn("Scanned PDF page 1", documents[0].page_content)
            self.assertTrue(documents[0].metadata["has_visual_content"])
            image_paths = documents[0].metadata["image_paths"]
            self.assertEqual(len(image_paths), 1)
            self.assertTrue(Path(image_paths[0]).exists())


class CompatibilityLayerTestCase(unittest.TestCase):
    """Verify legacy wrappers continue to work on top of EasyRAG."""

    def test_build_vector_index_search_docs_and_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents = [
                Document(
                    page_content="# Architecture\nRepoPilot uses LangChain for workflow orchestration.\n",
                    metadata={
                        "source_type": "doc",
                        "path": "docs/architecture.md",
                        "relative_path": "docs/architecture.md",
                        "title": "architecture",
                        "doc_id": "doc::architecture",
                    },
                )
            ]
            with mock.patch.dict(
                os.environ,
                {
                    "REPOPILOT_RAG_WORKING_DIR": tmp_dir,
                    "REPOPILOT_RAG_WORKSPACE": "compat",
                    "OPENAI_API_KEY": "",
                },
                clear=False,
            ):
                build_vector_index(documents)
                results = search_docs("How does workflow orchestration work?", k=3)
                tool_results = search_docs_tool.invoke({"query": "What uses LangChain?"})

            self.assertTrue(results)
            self.assertIn("LangChain", results[0].page_content)
            self.assertIn("architecture", tool_results)


class BuildIndexScriptTestCase(unittest.TestCase):
    """Verify the build script populates the EasyRAG workspace."""

    def test_build_index_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir) / "repo"
            docs_dir = repo_root / "docs"
            docs_dir.mkdir(parents=True)
            (docs_dir / "architecture.md").write_text(
                "# Architecture\nRepoPilot connects Streamlit, GitHub MCP, and EasyRAG.\n",
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {
                    "REPOPILOT_REPO_ROOT": str(repo_root),
                    "REPOPILOT_RAG_WORKING_DIR": str(repo_root / ".repopilot" / "rag_storage"),
                    "REPOPILOT_RAG_WORKSPACE": "demo",
                    "OPENAI_API_KEY": "",
                },
                clear=False,
            ):
                with contextlib.redirect_stdout(stdout):
                    build_index.main()

            output = stdout.getvalue()
            self.assertIn("documents=1", output)
            self.assertIn("workspace=demo", output)
            self.assertIn("vector_backend=fallback_token", output)
            self.assertTrue((repo_root / ".repopilot" / "rag_storage" / "demo" / "kv" / "documents.json").exists())


if __name__ == "__main__":
    unittest.main()
