"""Tests for agent integration with the EasyRAG subsystem."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from repopilot.agent import runner
from repopilot.rag import EasyRAG


def _stub_embedding(texts: list[str]) -> list[list[float]]:
    """Return deterministic embeddings for runner integration tests."""

    return [[float(text.lower().count("easyrag")), float(text.lower().count("langchain")), float(len(text.split()))] for text in texts]


def _stub_query_model(prompt: str, *, task: str, count: int = 1) -> str | list[str]:
    """Return deterministic rewrite and MQE outputs."""

    cleaned = prompt.split(":", 1)[-1].strip()
    if task == "rewrite":
        return f"{cleaned} repository docs"
    return [f"{cleaned} variant {index}" for index in range(1, count + 1)]


class RunnerEasyRAGIntegrationTestCase(unittest.TestCase):
    """Verify the fallback agent uses EasyRAG citations."""

    def test_invoke_agent_returns_rag_citations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rag = EasyRAG(
                working_dir=tmp_dir,
                workspace="runner",
                embedding_func=_stub_embedding,
                query_model_func=_stub_query_model,
            )
            runner._CACHED_AGENT = None
            runner._CACHED_RAG = None
            runner._CACHED_RAG_TOOL = None

            runner._run_async(rag.initialize_storages())
            try:
                runner._run_async(
                    rag.ainsert(
                        ["# Architecture\nRepoPilot uses EasyRAG and LangChain together.\n"],
                        ids=["doc::architecture"],
                        file_paths=["docs/architecture.md"],
                    )
                )

                with mock.patch.object(runner, "_get_rag", return_value=rag), mock.patch.object(
                    runner, "create_agent", None
                ), mock.patch.object(runner, "ChatOpenAI", None):
                    response = runner.invoke_agent(
                        user_query="How do EasyRAG and LangChain work together?",
                        user_id="user-1",
                        thread_id="thread-1",
                        memory_context=[],
                    )
            finally:
                runner._run_async(rag.finalize_storages())
                runner._CACHED_AGENT = None
                runner._CACHED_RAG = None
                runner._CACHED_RAG_TOOL = None

            self.assertTrue(response.citations)
            self.assertIn("architecture", response.citations[0].label.lower())
            self.assertIn("documentation", response.answer.lower())

    def test_invoke_agent_prefers_async_agent_path(self) -> None:
        class AsyncOnlyAgent:
            async def ainvoke(self, payload, config=None):
                del payload, config
                return {
                    "answer": "Async path worked.",
                    "citations": [],
                    "used_memory": [],
                    "confidence": "medium",
                }

        runner._CACHED_AGENT = None
        runner._CACHED_RAG = None
        runner._CACHED_RAG_TOOL = None
        with mock.patch.object(runner, "build_agent", return_value=AsyncOnlyAgent()):
            response = runner.invoke_agent(
                user_query="Does async invocation work?",
                user_id="user-1",
                thread_id="thread-async",
                memory_context=[],
            )

        self.assertEqual(response.answer, "Async path worked.")
        self.assertEqual(response.confidence, "medium")

    def test_invoke_agent_falls_back_to_docs_only_on_github_repo_404(self) -> None:
        class BrokenGitHubAgent:
            async def ainvoke(self, payload, config=None):
                del payload, config
                raise RuntimeError(
                    "failed to resolve git reference: failed to get repository info: "
                    "GET https://api.github.com/repos/supremewen/RepoPilot: 404 Not Found []"
                )

        docs_only = runner.FallbackAgent(tools=[], system_prompt="docs-only")

        with mock.patch.object(runner, "build_agent", return_value=BrokenGitHubAgent()), mock.patch.object(
            runner, "_build_docs_only_fallback", return_value=docs_only
        ), mock.patch.object(
            docs_only,
            "invoke",
            return_value={
                "answer": "I found matching documentation.",
                "citations": [],
                "used_memory": [],
                "confidence": "medium",
            },
        ):
            response = runner.invoke_agent(
                user_query="What does the doc say?",
                user_id="user-1",
                thread_id="thread-github-404",
                memory_context=[],
            )

        self.assertIn("GitHub evidence was unavailable", response.answer)
        self.assertIn("I found matching documentation.", response.answer)


if __name__ == "__main__":
    unittest.main()
