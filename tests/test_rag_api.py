"""Tests for the FastAPI RAG service surface."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from repopilot.service.api import create_app
from repopilot.rag import EasyRAG

_KEYWORDS = [
    "architecture",
    "langchain",
    "service",
    "runtime",
    "gateway",
    "api",
    "workflow",
]


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
        return cleaned
    if task == "mqe":
        return [f"{cleaned} variant {index}" for index in range(1, count + 1)]
    raise ValueError(task)


class RagApiTestCase(unittest.TestCase):
    """Verify the FastAPI service for queries, task status, and KG operations."""

    def _build_rag_factory(self, working_dir: str):
        def factory(workspace: str | None) -> EasyRAG:
            return EasyRAG(
                working_dir=working_dir,
                workspace=workspace or "default",
                embedding_func=_stub_embedding,
                query_model_func=_stub_query_model,
            )

        return factory

    def test_index_task_and_query_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir) / "repo"
            repo_root.mkdir()
            docs_dir = repo_root / "docs"
            docs_dir.mkdir()
            (docs_dir / "architecture.md").write_text(
                "# Architecture\nRepoPilot uses LangChain for repository architecture retrieval.\n",
                encoding="utf-8",
            )
            working_dir = Path(tmp_dir) / "rag"
            env = {
                "REPOPILOT_REPO_ROOT": str(repo_root),
                "REPOPILOT_RAG_WORKING_DIR": str(working_dir),
                "REPOPILOT_TASK_STATUS_PATH": str(Path(tmp_dir) / "task_status.json"),
                "REPOPILOT_RAG_WORKSPACE": "default",
            }
            with mock.patch.dict("os.environ", env, clear=False):
                app = create_app(rag_factory=self._build_rag_factory(str(working_dir)))
                with TestClient(app) as client:
                    response = client.post("/rag/index/tasks", json={"action": "full_sync", "workspace": "default"})
                    self.assertEqual(response.status_code, 202)
                    task_id = response.json()["task_id"]

                    deadline = time.time() + 5
                    payload = {}
                    while time.time() < deadline:
                        task_response = client.get(f"/rag/index/tasks/{task_id}")
                        payload = task_response.json()
                        if payload["status"] in {"succeeded", "failed"}:
                            break
                        time.sleep(0.05)

                    self.assertEqual(payload["status"], "succeeded")
                    query_response = client.post(
                        "/rag/query",
                        json={"query": "LangChain architecture", "mode": "naive", "workspace": "default", "top_k": 3, "chunk_top_k": 3},
                    )
                    self.assertEqual(query_response.status_code, 200)
                    query_payload = query_response.json()
                    self.assertTrue(query_payload["citations"])
                    self.assertIn("LangChain", query_payload["citations"][0]["snippet"])

    def test_kg_endpoints_update_queryable_entities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            working_dir = Path(tmp_dir) / "rag"
            env = {
                "REPOPILOT_RAG_WORKING_DIR": str(working_dir),
                "REPOPILOT_TASK_STATUS_PATH": str(Path(tmp_dir) / "task_status.json"),
                "REPOPILOT_RAG_WORKSPACE": "default",
            }
            with mock.patch.dict("os.environ", env, clear=False):
                app = create_app(rag_factory=self._build_rag_factory(str(working_dir)))
                with TestClient(app) as client:
                    service = client.post(
                        "/rag/kg/entities",
                        json={"id": "entity::service-layer", "label": "Service Layer", "entity_types": ["component"], "description": "Service boundary."},
                    ).json()
                    gateway = client.post(
                        "/rag/kg/entities",
                        json={"id": "entity::gateway-layer", "label": "Gateway Layer", "entity_types": ["component"], "description": "Gateway boundary."},
                    ).json()
                    client.post(
                        "/rag/kg/entities",
                        json={"id": "entity::runtime-config", "label": "Runtime Config", "entity_types": ["config"], "description": "Config source."},
                    )
                    relation_response = client.post(
                        "/rag/kg/relations",
                        json={
                            "id": "relation::service-runtime",
                            "source_entity_id": service["id"],
                            "target_entity_id": "entity::runtime-config",
                            "relation": "depends_on",
                            "description": "Service Layer depends on Runtime Config.",
                        },
                    )
                    self.assertEqual(relation_response.status_code, 200)

                    update_response = client.patch(
                        f"/rag/kg/entities/{service['id']}",
                        json={"label": "Runtime Layer"},
                    )
                    self.assertEqual(update_response.status_code, 200)

                    merge_response = client.post(
                        "/rag/kg/entities/merge",
                        json={"source_entity_id": service["id"], "target_entity_id": gateway["id"]},
                    )
                    self.assertEqual(merge_response.status_code, 200)

                    query_response = client.post(
                        "/rag/query",
                        json={"query": "Service Layer", "mode": "local", "workspace": "default", "top_k": 5, "chunk_top_k": 3},
                    )
                    self.assertEqual(query_response.status_code, 200)
                    payload = query_response.json()
                    self.assertIn("Gateway Layer", payload["entities"])
