"""Built-in single-node storage backends for RepoPilot's EasyRAG subsystem."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from repopilot.rag.base import BaseDocStatusStorage, BaseGraphStorage, BaseKVStorage, BaseVectorStorage

try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional dependency.
    nx = None

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Return lowercase lexical tokens for deterministic retrieval."""

    return _TOKEN_PATTERN.findall(text.lower())


def _read_json(path: Path, default: Any) -> Any:
    """Read JSON data if the file exists, otherwise return a default value."""

    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _write_json(path: Path, payload: Any) -> None:
    """Persist JSON using UTF-8 and stable pretty printing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class JSONKVStorage(BaseKVStorage):
    """Persist documents, chunks, summaries, and cache entries in JSON files."""

    def __init__(self, working_dir: str, workspace: str) -> None:
        super().__init__(working_dir, workspace)
        root = Path(working_dir) / workspace / "kv"
        self._root = root
        self._documents_path = root / "documents.json"
        self._chunks_path = root / "chunks.json"
        self._summaries_path = root / "summaries.json"
        self._cache_path = root / "cache.json"
        self._documents: dict[str, dict[str, Any]] = {}
        self._chunks: dict[str, dict[str, Any]] = {}
        self._summaries: dict[str, dict[str, Any]] = {}
        self._cache: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._documents = _read_json(self._documents_path, {})
        self._chunks = _read_json(self._chunks_path, {})
        self._summaries = _read_json(self._summaries_path, {})
        self._cache = _read_json(self._cache_path, {})

    async def finalize(self) -> None:
        _write_json(self._documents_path, self._documents)
        _write_json(self._chunks_path, self._chunks)
        _write_json(self._summaries_path, self._summaries)
        _write_json(self._cache_path, self._cache)

    async def upsert_documents(self, items: list[dict[str, Any]]) -> None:
        for item in items:
            self._documents[str(item["id"])] = item

    async def upsert_chunks(self, items: list[dict[str, Any]]) -> None:
        for item in items:
            self._chunks[str(item["id"])] = item

    async def upsert_summaries(self, items: list[dict[str, Any]]) -> None:
        for item in items:
            self._summaries[str(item["id"])] = item

    async def upsert_cache(self, namespace: str, key: str, value: Any) -> None:
        bucket = self._cache.setdefault(namespace, {})
        bucket[key] = value

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        return self._documents.get(document_id)

    async def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        return self._chunks.get(chunk_id)

    async def get_chunks(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        return [self._chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self._chunks]

    async def get_summary(self, summary_id: str) -> dict[str, Any] | None:
        return self._summaries.get(summary_id)

    async def get_stats(self) -> dict[str, Any]:
        chunk_strategy_counts: dict[str, int] = {}
        for chunk in self._chunks.values():
            strategy = str(chunk.get("metadata", {}).get("chunk_strategy", "unknown"))
            chunk_strategy_counts[strategy] = chunk_strategy_counts.get(strategy, 0) + 1
        pdf_documents = sum(1 for item in self._documents.values() if item.get("metadata", {}).get("source_type") == "pdf")
        return {
            "documents": len(self._documents),
            "chunks": len(self._chunks),
            "summaries": len(self._summaries),
            "cache_entries": sum(len(bucket) for bucket in self._cache.values()),
            "pdf_documents": pdf_documents,
            "chunk_strategy_counts": chunk_strategy_counts,
        }


class TokenVectorStorage(BaseVectorStorage):
    """Sparse token-overlap retrieval backend with namespace isolation."""

    def __init__(self, working_dir: str, workspace: str) -> None:
        super().__init__(working_dir, workspace)
        self._path = Path(working_dir) / workspace / "token_vector_store.json"
        self._store: dict[str, dict[str, dict[str, Any]]] = {}

    async def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._store = _read_json(self._path, {})

    async def finalize(self) -> None:
        _write_json(self._path, self._store)

    async def upsert(self, namespace: str, items: list[dict[str, Any]]) -> None:
        bucket = self._store.setdefault(namespace, {})
        for item in items:
            payload = dict(item)
            text = str(payload.get("text", ""))
            payload["tokens"] = payload.get("tokens") or _tokenize(text)
            bucket[str(payload["id"])] = payload

    async def similarity_search(self, namespace: str, query: str, top_k: int) -> list[dict[str, Any]]:
        bucket = self._store.get(namespace, {})
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        query_token_set = set(query_tokens)
        ranked: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
        for item in bucket.values():
            tokens = list(item.get("tokens", []))
            token_set = set(tokens)
            overlap = len(query_token_set & token_set)
            frequency = sum(tokens.count(token) for token in query_tokens)
            phrase_bonus = 1 if query.lower() in str(item.get("text", "")).lower() else 0
            if overlap == 0 and frequency == 0 and phrase_bonus == 0:
                continue
            ranked.append(((overlap, frequency, phrase_bonus), item))

        ranked.sort(key=lambda item: item[0], reverse=True)
        results: list[dict[str, Any]] = []
        for score, payload in ranked[:top_k]:
            item = dict(payload)
            item["score"] = score[0] * 100 + score[1] * 10 + score[2]
            item["vector_backend"] = "fallback_token"
            results.append(item)
        return results

    async def get_stats(self) -> dict[str, Any]:
        return {namespace: len(bucket) for namespace, bucket in self._store.items()}

    def get_backend_name(self) -> str:
        return "fallback_token"


class EmbeddingVectorStorage(BaseVectorStorage):
    """Dense embedding retrieval with token-overlap fallback."""

    def __init__(self, working_dir: str, workspace: str) -> None:
        super().__init__(working_dir, workspace)
        self._root = Path(working_dir) / workspace / "vector"
        self._records: dict[str, list[dict[str, Any]]] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._embedding_func = None
        self._token_fallback = TokenVectorStorage(working_dir, workspace)

    def set_embedding_func(self, embedding_func):
        self._embedding_func = embedding_func

    async def initialize(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        await self._token_fallback.initialize()
        for json_path in self._root.glob("*.json"):
            namespace = json_path.stem
            self._records[namespace] = _read_json(json_path, [])
            npy_path = self._root / f"{namespace}.npy"
            if npy_path.exists():
                try:
                    self._embeddings[namespace] = np.load(npy_path)
                except Exception:
                    self._embeddings[namespace] = np.zeros((0, 0), dtype=np.float32)

    async def finalize(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        for namespace, records in self._records.items():
            _write_json(self._root / f"{namespace}.json", records)
            embeddings = self._embeddings.get(namespace, np.zeros((0, 0), dtype=np.float32))
            np.save(self._root / f"{namespace}.npy", embeddings)
        await self._token_fallback.finalize()

    async def upsert(self, namespace: str, items: list[dict[str, Any]]) -> None:
        merged = {str(item["id"]): dict(item) for item in self._records.get(namespace, [])}
        for item in items:
            payload = dict(item)
            payload["tokens"] = payload.get("tokens") or _tokenize(str(payload.get("text", "")))
            merged[str(payload["id"])] = payload
        records = list(merged.values())
        self._records[namespace] = records
        await self._token_fallback.upsert(namespace, records)
        embeddings = await self._embed_records(records)
        self._embeddings[namespace] = embeddings

    async def _embed_records(self, records: list[dict[str, Any]]) -> np.ndarray:
        if not records or self._embedding_func is None:
            return np.zeros((0, 0), dtype=np.float32)
        try:
            inputs = [record.get("embedding_input", str(record.get("text", ""))) for record in records]
            values = self._embedding_func(inputs)
            embeddings = np.array(values, dtype=np.float32)
            if embeddings.ndim != 2 or len(embeddings) != len(records):
                return np.zeros((0, 0), dtype=np.float32)
            return embeddings
        except Exception:
            return np.zeros((0, 0), dtype=np.float32)

    async def similarity_search(self, namespace: str, query: str, top_k: int) -> list[dict[str, Any]]:
        records = self._records.get(namespace, [])
        embeddings = self._embeddings.get(namespace)
        if not records:
            return []

        if self._embedding_func is not None and embeddings is not None and embeddings.size:
            try:
                query_embedding = np.array(self._embedding_func([query]), dtype=np.float32)
                if query_embedding.ndim == 2 and query_embedding.shape[0] == 1 and query_embedding.shape[1] == embeddings.shape[1]:
                    normalized_records = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
                    normalized_query = query_embedding / np.maximum(np.linalg.norm(query_embedding, axis=1, keepdims=True), 1e-12)
                    scores = normalized_records @ normalized_query[0]
                    ranked_indices = np.argsort(scores)[::-1][:top_k]
                    results: list[dict[str, Any]] = []
                    for index in ranked_indices:
                        if float(scores[index]) <= 0:
                            continue
                        item = dict(records[int(index)])
                        item["score"] = float(scores[index])
                        item["vector_backend"] = "dense_embedding"
                        results.append(item)
                    if results:
                        return results
            except Exception:
                pass

        return await self._token_fallback.similarity_search(namespace, query, top_k)

    async def get_stats(self) -> dict[str, Any]:
        stats = {namespace: len(records) for namespace, records in self._records.items()}
        stats["vector_backend"] = self.get_backend_name()
        return stats

    def get_backend_name(self) -> str:
        for embeddings in self._embeddings.values():
            if embeddings.size:
                return "dense_embedding"
        return "fallback_token"


class NetworkXGraphStorage(BaseGraphStorage):
    """Graph backend that prefers networkx but falls back to plain dictionaries."""

    def __init__(self, working_dir: str, workspace: str) -> None:
        super().__init__(working_dir, workspace)
        self._path = Path(working_dir) / workspace / "graph.json"
        self._graph = nx.Graph() if nx is not None else {"nodes": {}, "edges": {}}

    async def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = _read_json(self._path, None)
        if not payload:
            return
        if nx is not None:
            self._graph = nx.node_link_graph(payload)
            return
        self._graph = payload

    async def finalize(self) -> None:
        if nx is not None:
            payload = nx.node_link_data(self._graph)
        else:
            payload = self._graph
        _write_json(self._path, payload)

    async def upsert_nodes(self, nodes: list[dict[str, Any]]) -> None:
        if nx is not None:
            for node in nodes:
                node_id = str(node["id"])
                existing = self._graph.nodes.get(node_id, {})
                merged = {**existing, **node}
                self._graph.add_node(node_id, **merged)
            return

        graph_nodes = self._graph.setdefault("nodes", {})
        for node in nodes:
            node_id = str(node["id"])
            graph_nodes[node_id] = {**graph_nodes.get(node_id, {}), **node}

    async def upsert_edges(self, edges: list[dict[str, Any]]) -> None:
        if nx is not None:
            for edge in edges:
                source = str(edge["source"])
                target = str(edge["target"])
                weight = float(edge.get("weight", 1.0))
                relation = str(edge.get("relation", "related_to"))
                if self._graph.has_edge(source, target):
                    data = self._graph[source][target]
                    relations = set(data.get("relations", []))
                    relations.add(relation)
                    self._graph[source][target]["weight"] = float(data.get("weight", 0.0)) + weight
                    self._graph[source][target]["relations"] = sorted(relations)
                else:
                    self._graph.add_edge(source, target, weight=weight, relations=[relation], metadata=edge.get("metadata", {}))
            return

        graph_nodes = self._graph.setdefault("nodes", {})
        graph_edges = self._graph.setdefault("edges", {})
        for edge in edges:
            source = str(edge["source"])
            target = str(edge["target"])
            graph_nodes.setdefault(source, {"id": source})
            graph_nodes.setdefault(target, {"id": target})
            key = "|".join(sorted((source, target)))
            existing = graph_edges.get(key, {"source": source, "target": target, "weight": 0.0, "relations": []})
            relations = set(existing.get("relations", []))
            relations.add(str(edge.get("relation", "related_to")))
            existing["weight"] = float(existing.get("weight", 0.0)) + float(edge.get("weight", 1.0))
            existing["relations"] = sorted(relations)
            existing["metadata"] = edge.get("metadata", {})
            graph_edges[key] = existing

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        if nx is not None:
            if node_id not in self._graph:
                return None
            return {"id": node_id, **dict(self._graph.nodes[node_id])}
        node = self._graph.get("nodes", {}).get(node_id)
        return dict(node) if node else None

    async def get_neighbors(
        self,
        node_ids: list[str],
        *,
        kind_filter: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = {}
        if nx is not None:
            for node_id in node_ids:
                if node_id not in self._graph:
                    continue
                for neighbor_id in self._graph.neighbors(node_id):
                    if neighbor_id in node_ids:
                        continue
                    data = dict(self._graph.nodes[neighbor_id])
                    if kind_filter and data.get("kind") != kind_filter:
                        continue
                    edge = self._graph[node_id][neighbor_id]
                    scores[neighbor_id] = scores.get(neighbor_id, 0.0) + float(edge.get("weight", 1.0))
            ranked_ids = sorted(scores, key=lambda item: scores[item], reverse=True)[:limit]
            return [{"id": node_id, **dict(self._graph.nodes[node_id]), "score": scores[node_id]} for node_id in ranked_ids]

        graph_nodes = self._graph.get("nodes", {})
        graph_edges = self._graph.get("edges", {})
        for edge in graph_edges.values():
            source = edge["source"]
            target = edge["target"]
            for node_id, neighbor_id in ((source, target), (target, source)):
                if node_id not in node_ids or neighbor_id in node_ids:
                    continue
                node = graph_nodes.get(neighbor_id, {})
                if kind_filter and node.get("kind") != kind_filter:
                    continue
                scores[neighbor_id] = scores.get(neighbor_id, 0.0) + float(edge.get("weight", 1.0))
        ranked_ids = sorted(scores, key=lambda item: scores[item], reverse=True)[:limit]
        return [{**graph_nodes[node_id], "score": scores[node_id]} for node_id in ranked_ids]

    async def top_nodes(self, *, kind: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        if nx is not None:
            ranked: list[tuple[int, str]] = []
            for node_id, data in self._graph.nodes(data=True):
                if kind and data.get("kind") != kind:
                    continue
                ranked.append((int(self._graph.degree(node_id)), node_id))
            ranked.sort(reverse=True)
            return [
                {"id": node_id, **dict(self._graph.nodes[node_id]), "score": degree}
                for degree, node_id in ranked[:limit]
            ]

        graph_nodes = self._graph.get("nodes", {})
        graph_edges = self._graph.get("edges", {})
        degree_map = {node_id: 0 for node_id in graph_nodes}
        for edge in graph_edges.values():
            degree_map[edge["source"]] = degree_map.get(edge["source"], 0) + 1
            degree_map[edge["target"]] = degree_map.get(edge["target"], 0) + 1
        ranked: list[tuple[int, str]] = []
        for node_id, data in graph_nodes.items():
            if kind and data.get("kind") != kind:
                continue
            ranked.append((degree_map.get(node_id, 0), node_id))
        ranked.sort(reverse=True)
        return [{**graph_nodes[node_id], "score": degree} for degree, node_id in ranked[:limit]]

    async def get_stats(self) -> dict[str, Any]:
        if nx is not None:
            return {"nodes": int(self._graph.number_of_nodes()), "edges": int(self._graph.number_of_edges())}
        return {"nodes": len(self._graph.get("nodes", {})), "edges": len(self._graph.get("edges", {}))}


class JSONDocStatusStorage(BaseDocStatusStorage):
    """Persist document processing status as JSON."""

    def __init__(self, working_dir: str, workspace: str) -> None:
        super().__init__(working_dir, workspace)
        self._path = Path(working_dir) / workspace / "doc_status.json"
        self._statuses: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._statuses = _read_json(self._path, {})

    async def finalize(self) -> None:
        _write_json(self._path, self._statuses)

    async def mark_status(
        self,
        document_id: str,
        status: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._statuses[document_id] = {
            "document_id": document_id,
            "status": status,
            "metadata": metadata or {},
        }

    async def get_status(self, document_id: str) -> dict[str, Any] | None:
        return self._statuses.get(document_id)

    async def list_statuses(self) -> list[dict[str, Any]]:
        return list(self._statuses.values())

    async def get_stats(self) -> dict[str, Any]:
        return {"documents": len(self._statuses)}


__all__ = [
    "EmbeddingVectorStorage",
    "JSONDocStatusStorage",
    "JSONKVStorage",
    "NetworkXGraphStorage",
    "TokenVectorStorage",
]
