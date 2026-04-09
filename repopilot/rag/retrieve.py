"""Retrieval, fusion, hydration, and rerank helpers for EasyRAG queries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from repopilot.compat import Document
from repopilot.rag.base import QueryParam, QueryResult
from repopilot.rag.utils import dedupe_strings, extract_entity_candidates, slugify

if TYPE_CHECKING:
    from repopilot.rag.easyrag import EasyRAG


def merge_ranked_records(rank_groups: list[tuple[float, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    """Merge ranked record lists by accumulating weighted scores."""

    merged: dict[str, dict[str, Any]] = {}
    for weight, records in rank_groups:
        for position, record in enumerate(records):
            record_id = str(record["id"])
            candidate = merged.setdefault(record_id, dict(record))
            score = float(record.get("score", len(records) - position))
            candidate["score"] = float(candidate.get("score", 0.0)) + score * weight
            if "vector_backend" not in candidate and "vector_backend" in record:
                candidate["vector_backend"] = record["vector_backend"]
    return sorted(merged.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)


def rrf_fuse(record_groups: list[list[dict[str, Any]]], *, k: int = 60) -> list[dict[str, Any]]:
    """Fuse multiple ranked lists with reciprocal rank fusion."""

    merged: dict[str, dict[str, Any]] = {}
    for records in record_groups:
        for rank, record in enumerate(records, start=1):
            record_id = str(record["id"])
            candidate = merged.setdefault(record_id, dict(record))
            candidate["score"] = float(candidate.get("score", 0.0)) + 1.0 / (k + rank)
            if "vector_backend" not in candidate and "vector_backend" in record:
                candidate["vector_backend"] = record["vector_backend"]
    return sorted(merged.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)


async def hydrate_record(rag: "EasyRAG", record: dict[str, Any]) -> dict[str, Any] | None:
    """Hydrate a ranked record with full text and metadata for citations/rerank."""

    record_id = str(record["id"])
    if "::chunk::" in record_id:
        chunk = await rag.kv_storage.get_chunk(record_id)
        if chunk is None:
            return None
        return {
            "id": record_id,
            "text": str(chunk.get("text", "")),
            "title": str(chunk.get("title", "")),
            "path": str(chunk.get("path", "")),
            "metadata": dict(chunk.get("metadata", {})),
            "score": float(record.get("score", 0.0)),
            "vector_backend": str(record.get("vector_backend", "")),
        }
    if record_id.startswith("summary::"):
        summary = await rag.kv_storage.get_summary(record_id)
        if summary is None:
            return None
        return {
            "id": record_id,
            "text": str(summary.get("text", "")),
            "title": str(summary.get("title", "")),
            "path": str(summary.get("path", "")),
            "metadata": dict(summary.get("metadata", {})),
            "score": float(record.get("score", 0.0)),
            "vector_backend": str(record.get("vector_backend", "")),
        }
    return dict(record)


async def hydrate_records(rag: "EasyRAG", records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hydrate ranked records in order."""

    hydrated: list[dict[str, Any]] = []
    for record in records:
        item = await hydrate_record(rag, record)
        if item is not None:
            hydrated.append(item)
    return hydrated


async def chunks_to_documents(records: list[dict[str, Any]]) -> list[Document]:
    """Convert hydrated records into LangChain-compatible documents."""

    return [
        Document(page_content=str(record.get("text", "")), metadata=dict(record.get("metadata", {})))
        for record in records
        if str(record.get("text", "")).strip()
    ]


async def naive_query(rag: "EasyRAG", query: str, param: QueryParam) -> list[dict[str, Any]]:
    """Retrieve chunks directly from the chunk vector namespace."""

    return await rag.vector_storage.similarity_search("chunk", query, param.chunk_top_k)


async def local_query(rag: "EasyRAG", query: str, param: QueryParam) -> tuple[list[dict[str, Any]], list[str]]:
    """Retrieve chunks related to query-time entities plus dense backfill."""

    extracted_entities = extract_entity_candidates(query, {"title": query, "path": ""})
    entity_hits = await rag.vector_storage.similarity_search("entity", query, param.top_k)
    entity_ids = [f"entity::{slugify(entity)}" for entity in extracted_entities]
    entity_ids.extend(str(item["id"]) for item in entity_hits if str(item["id"]) not in entity_ids)
    entities = dedupe_strings(extracted_entities + [str(item.get("metadata", {}).get("label", item.get("text", ""))) for item in entity_hits])
    neighbors = await rag.graph_storage.get_neighbors(entity_ids, kind_filter="chunk", limit=param.chunk_top_k)
    dense_backfill = await rag.vector_storage.similarity_search("chunk", query, param.chunk_top_k)
    selected = merge_ranked_records([(1.0, neighbors), (0.6, dense_backfill)])
    return selected, entities


async def global_query(rag: "EasyRAG", query: str, param: QueryParam) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retrieve broad context from summaries, central entities, and dense backfill."""

    summary_hits = await rag.vector_storage.similarity_search("summary", query, param.top_k)
    central_entities = await rag.graph_storage.top_nodes(kind="entity", limit=param.top_k)
    central_neighbors = await rag.graph_storage.get_neighbors(
        [str(item["id"]) for item in central_entities],
        kind_filter="chunk",
        limit=param.chunk_top_k,
    )
    dense_backfill = await rag.vector_storage.similarity_search("chunk", query, param.chunk_top_k)
    merged = merge_ranked_records([(1.0, summary_hits), (0.7, central_neighbors), (0.4, dense_backfill)])
    return merged, central_entities


def trim_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep the leading ranked records."""

    return records[: max(limit, 0)]


def combine_mode_results(param: QueryParam, *groups: tuple[float, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Combine mode-level result groups using the requested fusion strategy."""

    records = [group for _, group in groups]
    if param.retrieval_fusion == "rrf":
        return rrf_fuse(records)
    return merge_ranked_records(list(groups))


async def run_variant_queries(
    rag: "EasyRAG",
    retrieval_queries: list[str],
    param: QueryParam,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Run naive/local/global retrieval for each prepared query variant."""

    naive_groups: list[list[dict[str, Any]]] = []
    local_groups: list[list[dict[str, Any]]] = []
    global_groups: list[list[dict[str, Any]]] = []
    relation_groups: list[list[dict[str, Any]]] = []
    entities: list[str] = []
    central_entities: list[dict[str, Any]] = []

    for retrieval_query in retrieval_queries:
        naive_groups.append(await naive_query(rag, retrieval_query, param))
        local_hits, local_entities = await local_query(rag, retrieval_query, param)
        global_hits, global_central = await global_query(rag, retrieval_query, param)
        local_groups.append(local_hits)
        global_groups.append(global_hits)
        relation_groups.append(await rag.vector_storage.similarity_search("relation", retrieval_query, param.top_k))
        entities.extend(local_entities)
        if not central_entities:
            central_entities = global_central

    return (
        rrf_fuse(naive_groups),
        rrf_fuse(local_groups),
        rrf_fuse(global_groups),
        dedupe_strings(entities),
        rrf_fuse(relation_groups),
    )


async def execute_query(rag: "EasyRAG", query: str, param: QueryParam) -> QueryResult:
    """Execute one multi-mode query over the active RAG workspace."""

    mode = param.mode.lower().strip()
    prepared = rag.query_preprocessor.prepare(query, param)
    naive_hits, local_hits, global_hits, local_entities, relation_hits = await run_variant_queries(
        rag,
        prepared.retrieval_queries,
        param,
    )

    if mode == "naive":
        selected = naive_hits
    elif mode == "local":
        selected = local_hits
    elif mode == "global":
        selected = global_hits
    elif mode == "hybrid":
        selected = combine_mode_results(param, (1.0, local_hits), (1.0, global_hits))
    elif mode == "mix":
        selected = combine_mode_results(param, (1.0, local_hits), (1.0, global_hits), (0.85, naive_hits))
    else:
        raise ValueError(f"Unsupported query mode: {param.mode}")

    hydrated = await hydrate_records(rag, trim_records(selected, param.chunk_top_k * 3))
    rerank_applied = False
    if mode == "mix" and rag.reranker_func is not None:
        try:
            hydrated = list(rag.reranker_func(prepared.rewritten_query, hydrated))
            rerank_applied = True
        except Exception:
            rerank_applied = False
    elif mode == "hybrid" and param.enable_rerank and rag.reranker_func is not None:
        try:
            hydrated = list(rag.reranker_func(prepared.rewritten_query, hydrated))
            rerank_applied = True
        except Exception:
            rerank_applied = False

    hydrated = trim_records(hydrated, param.chunk_top_k)
    chunks = await chunks_to_documents(hydrated)
    citations = [
        {
            "source_type": str(document.metadata.get("source_type", "doc")),
            "title": str(document.metadata.get("title", "Document")),
            "location": str(document.metadata.get("path", "")),
            "snippet": document.page_content[:400].strip(),
        }
        for document in chunks
    ]
    hit_chunk_strategies = sorted(
        {
            str(document.metadata.get("chunk_strategy", "unknown"))
            for document in chunks
            if document.metadata.get("chunk_strategy")
        }
    )
    vector_backend = "dense_embedding" if any(str(item.get("vector_backend")) == "dense_embedding" for item in hydrated) else "fallback_token"

    return QueryResult(
        mode=mode,
        chunks=chunks,
        citations=citations,
        entities=local_entities[: param.top_k],
        relations=[
            {
                "id": str(item["id"]),
                "snippet": str(item.get("text", ""))[:200],
                "score": float(item.get("score", 0.0)),
            }
            for item in relation_hits[: param.top_k]
        ],
        metadata={
            "workspace": rag.workspace,
            "working_dir": str(rag.workspace_dir),
            "selected_count": len(citations),
            "original_query": prepared.original_query,
            "rewritten_query": prepared.rewritten_query,
            "expanded_queries": prepared.expanded_queries,
            "retrieval_queries": prepared.retrieval_queries,
            "retrieval_mode": mode,
            "hit_chunk_strategies": hit_chunk_strategies,
            "vector_backend": vector_backend,
            "rerank_applied": rerank_applied,
        },
    )
