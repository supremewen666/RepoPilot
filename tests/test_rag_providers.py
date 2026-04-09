"""Tests for RepoPilot's default provider adapters."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from repopilot.rag import providers


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._body


class _FakeHTTPX:
    def __init__(self, response_body):
        self.calls = []
        self._response_body = response_body

    def post(self, url, json, headers, timeout):
        self.calls.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return _FakeResponse(self._response_body)


class _FakeEmbeddingsAPI:
    def __init__(self):
        self.calls = []

    def create(self, *, model, input):
        self.calls.append({"model": model, "input": input})
        return type(
            "EmbeddingsResponse",
            (),
            {"data": [type("EmbeddingItem", (), {"embedding": [0.1, 0.2]})()]},
        )()


class _FakeOpenAIClient:
    instances = []

    def __init__(self, *, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.embeddings = _FakeEmbeddingsAPI()
        _FakeOpenAIClient.instances.append(self)


class ProviderAdapterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.env_backup = dict(os.environ)
        os.environ["OPENAI_API_KEY"] = "test-key"

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.env_backup)
        _FakeOpenAIClient.instances.clear()

    def test_dashscope_vl_embedding_uses_official_endpoint(self) -> None:
        os.environ["REPOPILOT_EMBEDDING_MODEL_NAME"] = "Qwen3-VL-Embedding-4B"
        os.environ["REPOPILOT_EMBEDDING_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        image_path = os.path.join(os.getcwd(), "tmp-test-image.png")
        with open(image_path, "wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\n")
        fake_httpx = _FakeHTTPX(
            {
                "output": {
                    "embeddings": [
                        {"embedding": [1.0, 2.0]},
                        {"embedding": [3.0, 4.0]},
                    ]
                }
            }
        )
        try:
            with patch.object(providers, "httpx", fake_httpx):
                vectors = providers.default_embedding_func(["alpha", {"text": "beta", "image_paths": [image_path]}])
        finally:
            os.remove(image_path)

        self.assertEqual(vectors, [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(
            fake_httpx.calls[0]["url"],
            "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
        )
        self.assertEqual(fake_httpx.calls[0]["json"]["input"]["contents"][0], {"text": "alpha"})
        self.assertEqual(fake_httpx.calls[0]["json"]["input"]["contents"][1]["text"], "beta")
        self.assertTrue(fake_httpx.calls[0]["json"]["input"]["contents"][1]["image"].startswith("data:image/png;base64,"))

    def test_openai_compatible_embedding_path_still_works(self) -> None:
        os.environ["REPOPILOT_EMBEDDING_MODEL_NAME"] = "text-embedding-v4"
        os.environ["REPOPILOT_EMBEDDING_BASE_URL"] = "https://api.example.com/v1"

        with patch.object(providers, "OpenAI", _FakeOpenAIClient):
            vectors = providers.default_embedding_func(["alpha"])

        self.assertEqual(vectors, [[0.1, 0.2]])
        self.assertEqual(len(_FakeOpenAIClient.instances), 1)
        self.assertEqual(_FakeOpenAIClient.instances[0].base_url, "https://api.example.com/v1")
        self.assertEqual(
            _FakeOpenAIClient.instances[0].embeddings.calls[0],
            {"model": "text-embedding-v4", "input": ["alpha"]},
        )

    def test_dashscope_vl_rerank_uses_official_endpoint(self) -> None:
        os.environ["REPOPILOT_RERANK_MODEL_NAME"] = "Qwen3-VL-Reranker-8B"
        os.environ["REPOPILOT_RERANK_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        image_path = os.path.join(os.getcwd(), "tmp-test-image.png")
        with open(image_path, "wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\n")
        fake_httpx = _FakeHTTPX(
            {
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.3},
                    ]
                }
            }
        )

        try:
            with patch.object(providers, "httpx", fake_httpx):
                ranked = providers.default_reranker_func(
                    "find auth code",
                    [{"text": "first"}, {"text": "second", "metadata": {"image_paths": [image_path]}}],
                )
        finally:
            os.remove(image_path)

        self.assertEqual([item["text"] for item in ranked], ["second", "first"])
        self.assertEqual(
            fake_httpx.calls[0]["url"],
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        )
        self.assertEqual(
            fake_httpx.calls[0]["json"]["input"]["query"],
            {"text": "find auth code"},
        )
        self.assertEqual(fake_httpx.calls[0]["json"]["input"]["documents"][0], {"text": "first"})
        self.assertEqual(fake_httpx.calls[0]["json"]["input"]["documents"][1]["text"], "second")
        self.assertTrue(fake_httpx.calls[0]["json"]["input"]["documents"][1]["image"].startswith("data:image/png;base64,"))

    def test_dashscope_intl_text_rerank_uses_compatible_endpoint(self) -> None:
        os.environ["REPOPILOT_RERANK_MODEL_NAME"] = "qwen3-rerank"
        os.environ["REPOPILOT_RERANK_BASE_URL"] = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        fake_httpx = _FakeHTTPX({"results": [{"index": 0, "relevance_score": 0.7}]})

        with patch.object(providers, "httpx", fake_httpx):
            providers.default_reranker_func("find auth code", [{"text": "first"}])

        self.assertEqual(
            fake_httpx.calls[0]["url"],
            "https://dashscope-intl.aliyuncs.com/compatible-api/v1/reranks",
        )


if __name__ == "__main__":
    unittest.main()
