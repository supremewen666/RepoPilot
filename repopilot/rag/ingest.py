"""Document-to-storage ingestion helpers for EasyRAG."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from repopilot.compat import Document
from repopilot.rag.documents import chunk_documents
from repopilot.rag.utils import extract_entity_candidates, slugify, summarize_document

if TYPE_CHECKING:
    from repopilot.rag.easyrag import EasyRAG


def _build_embedding_input(text: str, metadata: dict[str, Any]) -> str | dict[str, Any]:
    """Build embedding input that can preserve image assets for VL models."""

    image_paths = [str(path) for path in metadata.get("image_paths", []) if str(path).strip()]
    if not image_paths:
        return text
    return {
        "text": text,
        "image_paths": image_paths,
    }


def build_insert_payloads(
    rag: "EasyRAG",
    documents: list[Document],
    *,
    chunk_strategy_override: str | None = None,
) -> dict[str, Any]:
    """Transform documents into storage payloads."""

    payloads: dict[str, Any] = {
        "documents": [],
        "chunks": [],
        "summaries": [],
        "graph_nodes": [],
        "graph_edges": [],
        "vector_chunks": [],
        "vector_entities": [],
        "vector_relations": [],
        "vector_summaries": [],
        "statuses": [],
        "chunk_strategy_counts": {},
    }
    entity_labels: dict[str, str] = {}
    relation_ids: set[str] = set()
    pdf_documents = 0

    all_chunks = chunk_documents(
        documents,
        config=rag.chunking_config,
        chunk_strategy_override=chunk_strategy_override,
        rag=rag,
    )
    chunks_by_doc: dict[str, list[Document]] = {}
    for chunk in all_chunks:
        chunks_by_doc.setdefault(str(chunk.metadata["doc_id"]), []).append(chunk)

    for document in documents:
        doc_id = str(document.metadata["doc_id"])
        path = str(document.metadata.get("path", ""))
        title = str(document.metadata.get("title", "Document"))
        summary_id = f"summary::{doc_id}"
        summary_text = summarize_document(document.page_content)
        source_type = str(document.metadata.get("source_type", "doc"))
        if source_type == "pdf":
            pdf_documents += 1

        payloads["documents"].append(
            {
                "id": doc_id,
                "title": title,
                "path": path,
                "relative_path": str(document.metadata.get("relative_path", path)),
                "text": document.page_content,
                "summary_id": summary_id,
                "metadata": dict(document.metadata),
            }
        )
        payloads["summaries"].append(
            {
                "id": summary_id,
                "document_id": doc_id,
                "title": f"{title} summary",
                "path": path,
                "text": summary_text,
                "metadata": {"source_type": "summary", "doc_id": doc_id, "path": path, "title": title},
            }
        )
        payloads["graph_nodes"].extend(
            [
                {"id": doc_id, "kind": "document", "label": title, "path": path},
                {"id": summary_id, "kind": "summary", "label": f"{title} summary", "path": path, "text": summary_text},
            ]
        )
        payloads["graph_edges"].append({"source": doc_id, "target": summary_id, "relation": "document_summary", "weight": 1.0})
        payloads["vector_summaries"].append(
            {
                "id": summary_id,
                "text": summary_text,
                "metadata": {"doc_id": doc_id, "title": title, "path": path, "kind": "summary"},
            }
        )

        chunk_entities_by_id: dict[str, list[str]] = {}
        for chunk in chunks_by_doc.get(doc_id, []):
            chunk_id = str(chunk.metadata["chunk_uid"])
            strategy = str(chunk.metadata.get("chunk_strategy", "unknown"))
            payloads["chunk_strategy_counts"][strategy] = payloads["chunk_strategy_counts"].get(strategy, 0) + 1
            chunk_record = {
                "id": chunk_id,
                "document_id": doc_id,
                "text": chunk.page_content,
                "title": title,
                "path": path,
                "metadata": dict(chunk.metadata),
            }
            payloads["chunks"].append(chunk_record)
            payloads["graph_nodes"].append(
                {
                    "id": chunk_id,
                    "kind": "chunk",
                    "label": f"{title} chunk {chunk.metadata.get('chunk_id', 0)}",
                    "path": path,
                    "text": chunk.page_content[:240],
                    "chunk_strategy": strategy,
                }
            )
            payloads["graph_edges"].append({"source": doc_id, "target": chunk_id, "relation": "document_chunk", "weight": 1.0})
            payloads["vector_chunks"].append(
                {
                    "id": chunk_id,
                    "text": chunk.page_content,
                    "embedding_input": _build_embedding_input(chunk.page_content, dict(chunk.metadata)),
                    "metadata": {
                        "doc_id": doc_id,
                        "title": title,
                        "path": path,
                        "kind": "chunk",
                        "chunk_strategy": strategy,
                        "page_number": chunk.metadata.get("page_number"),
                        "image_paths": list(chunk.metadata.get("image_paths", []) or []),
                    },
                }
            )

            entities = extract_entity_candidates(chunk.page_content, dict(chunk.metadata))
            chunk_entities_by_id[chunk_id] = entities
            for entity in entities:
                entity_id = f"entity::{slugify(entity)}"
                entity_labels[entity_id] = entity
                payloads["graph_nodes"].append({"id": entity_id, "kind": "entity", "label": entity})
                payloads["graph_edges"].append({"source": entity_id, "target": doc_id, "relation": "entity_document", "weight": 1.0})
                payloads["graph_edges"].append({"source": entity_id, "target": chunk_id, "relation": "entity_chunk", "weight": 1.0})

            for left_index, left_entity in enumerate(entities):
                for right_entity in entities[left_index + 1 :]:
                    left_id = f"entity::{slugify(left_entity)}"
                    right_id = f"entity::{slugify(right_entity)}"
                    relation_id = "::".join(sorted((left_id, right_id)) + [chunk_id])
                    if relation_id in relation_ids:
                        continue
                    relation_ids.add(relation_id)
                    payloads["graph_edges"].append(
                        {"source": left_id, "target": right_id, "relation": "entity_cooccurs", "weight": 0.5, "metadata": {"chunk_id": chunk_id}}
                    )
                    payloads["vector_relations"].append(
                        {
                            "id": f"relation::{slugify(relation_id)}",
                            "text": f"{left_entity} co-occurs with {right_entity} in {title}",
                            "metadata": {"left": left_id, "right": right_id, "chunk_id": chunk_id, "path": path, "kind": "relation"},
                        }
                    )

        payloads["statuses"].append({"document_id": doc_id, "status": "indexed", "metadata": {"path": path, "chunk_count": len(chunk_entities_by_id)}})

    for entity_id, label in entity_labels.items():
        payloads["vector_entities"].append(
            {
                "id": entity_id,
                "text": label,
                "metadata": {"label": label, "kind": "entity"},
            }
        )

    payloads["pdf_documents"] = pdf_documents
    return payloads


async def ingest_documents(rag: "EasyRAG", documents: list[Document]) -> dict[str, int]:
    """Insert documents into the EasyRAG workspace."""

    payloads = build_insert_payloads(rag, documents)
    await rag.kv_storage.upsert_documents(payloads["documents"])
    await rag.kv_storage.upsert_chunks(payloads["chunks"])
    await rag.kv_storage.upsert_summaries(payloads["summaries"])
    await rag.vector_storage.upsert("chunk", payloads["vector_chunks"])
    await rag.vector_storage.upsert("entity", payloads["vector_entities"])
    await rag.vector_storage.upsert("relation", payloads["vector_relations"])
    await rag.vector_storage.upsert("summary", payloads["vector_summaries"])
    await rag.graph_storage.upsert_nodes(payloads["graph_nodes"])
    await rag.graph_storage.upsert_edges(payloads["graph_edges"])
    for status in payloads["statuses"]:
        await rag.doc_status_storage.mark_status(status["document_id"], status["status"], metadata=status["metadata"])
    return {
        "documents": len(payloads["documents"]),
        "chunks": len(payloads["chunks"]),
        "entities": len(payloads["vector_entities"]),
        "relations": len(payloads["vector_relations"]),
        "pdf_documents": payloads["pdf_documents"],
    }
