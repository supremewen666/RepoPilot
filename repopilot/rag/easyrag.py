"""EasyRAG-style orchestrator for RepoPilot's single-repository knowledge subsystem."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from repopilot.compat import Document
from repopilot.config import get_rag_working_dir, get_rag_workspace
from repopilot.rag.base import (
    BaseDocStatusStorage,
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    ChunkerFunc,
    EmbeddingFunc,
    LLMFunc,
    QueryModelFunc,
    QueryParam,
    QueryResult,
    RerankerFunc,
)
from repopilot.rag.chunking import ChunkingConfig, build_chunker_registry
from repopilot.rag.kg import EmbeddingVectorStorage, JSONDocStatusStorage, JSONKVStorage, NetworkXGraphStorage
from repopilot.rag.operate import execute_query, ingest_documents, prepare_documents_for_insert
from repopilot.rag.preprocess import QueryPreprocessor
from repopilot.rag.providers import (
    can_use_openai_compatible_models,
    default_embedding_func,
    default_query_model_func,
    default_reranker_func,
)


class EasyRAG:
    """Single-repository RAG orchestrator with pluggable storage backends."""

    def __init__(
        self,
        *,
        working_dir: str | Path | None = None,
        workspace: str | None = None,
        llm_model_func: LLMFunc | None = None,
        query_model_func: QueryModelFunc | None = None,
        embedding_func: EmbeddingFunc | None = None,
        reranker_func: RerankerFunc | None = None,
        kv_storage_cls: type[BaseKVStorage] = JSONKVStorage,
        vector_storage_cls: type[BaseVectorStorage] = EmbeddingVectorStorage,
        graph_storage_cls: type[BaseGraphStorage] = NetworkXGraphStorage,
        doc_status_storage_cls: type[BaseDocStatusStorage] = JSONDocStatusStorage,
        chunking_config: ChunkingConfig | None = None,
        chunker_registry: dict[str, ChunkerFunc] | None = None,
    ) -> None:
        self.working_dir = Path(working_dir or get_rag_working_dir()).resolve()
        self.workspace = workspace or get_rag_workspace()
        self.workspace_dir = self.working_dir / self.workspace
        self.llm_model_func = llm_model_func
        self.query_model_func = query_model_func or (default_query_model_func if can_use_openai_compatible_models() else None)
        self.embedding_func = embedding_func or (default_embedding_func if can_use_openai_compatible_models() else None)
        self.reranker_func = reranker_func or (default_reranker_func if can_use_openai_compatible_models() else None)
        self.chunking_config = chunking_config or ChunkingConfig()
        self.chunker_registry = chunker_registry or build_chunker_registry()
        self.query_preprocessor = QueryPreprocessor(self.query_model_func)
        self.kv_storage = kv_storage_cls(str(self.working_dir), self.workspace)
        self.vector_storage = vector_storage_cls(str(self.working_dir), self.workspace)
        self.graph_storage = graph_storage_cls(str(self.working_dir), self.workspace)
        self.doc_status_storage = doc_status_storage_cls(str(self.working_dir), self.workspace)
        self.vector_storage.set_embedding_func(self.embedding_func)
        self._initialized = False

    async def initialize_storages(self) -> None:
        """Load or create all configured storages."""

        if self._initialized:
            return
        await self.kv_storage.initialize()
        await self.vector_storage.initialize()
        await self.graph_storage.initialize()
        await self.doc_status_storage.initialize()
        self._initialized = True

    async def finalize_storages(self) -> None:
        """Persist all configured storages."""

        if not self._initialized:
            return
        await self.kv_storage.finalize()
        await self.vector_storage.finalize()
        await self.graph_storage.finalize()
        await self.doc_status_storage.finalize()
        self._initialized = False

    async def ainsert(
        self,
        texts: str | Sequence[str],
        *,
        ids: Sequence[str] | None = None,
        file_paths: Sequence[str] | None = None,
    ) -> dict[str, int]:
        """Insert one or more documents into the configured workspace."""

        self._ensure_initialized()
        documents = prepare_documents_for_insert(texts, ids=ids, file_paths=file_paths)
        return await ingest_documents(self, documents)

    async def ainsert_documents(self, documents: Sequence[Document]) -> dict[str, int]:
        """Insert pre-built Document objects while preserving their metadata."""

        self._ensure_initialized()
        return await ingest_documents(self, list(documents))

    async def aquery(self, query: str, param: QueryParam) -> QueryResult:
        """Run one multi-mode knowledge query against the active workspace."""

        self._ensure_initialized()
        return await execute_query(self, query, param)

    async def get_stats(self) -> dict[str, int]:
        """Return aggregated storage stats for diagnostics and build output."""

        self._ensure_initialized()
        stats = await self.kv_storage.get_stats()
        vector_stats = await self.vector_storage.get_stats()
        graph_stats = await self.graph_storage.get_stats()
        status_stats = await self.doc_status_storage.get_stats()
        return {
            **stats,
            "entity_vectors": int(vector_stats.get("entity", 0)),
            "relation_vectors": int(vector_stats.get("relation", 0)),
            "chunk_vectors": int(vector_stats.get("chunk", 0)),
            "summary_vectors": int(vector_stats.get("summary", 0)),
            "vector_backend": str(vector_stats.get("vector_backend", self.vector_storage.get_backend_name())),
            "graph_nodes": int(graph_stats.get("nodes", 0)),
            "graph_edges": int(graph_stats.get("edges", 0)),
            "status_records": int(status_stats.get("documents", 0)),
        }

    def can_rerank(self) -> bool:
        """Return whether a reranker is configured."""

        return self.reranker_func is not None and can_use_openai_compatible_models()

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("EasyRAG storages are not initialized. Call await rag.initialize_storages() first.")
