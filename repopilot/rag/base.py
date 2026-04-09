"""Core types and storage abstractions for RepoPilot's EasyRAG-style subsystem."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from repopilot.compat import Document

QueryMode = str


@dataclass(frozen=True)
class QueryParam:
    """Configuration for one repository knowledge query."""

    mode: QueryMode = "hybrid"
    top_k: int = 8
    chunk_top_k: int = 6
    max_entity_tokens: int = 6000
    max_relation_tokens: int = 8000
    max_total_tokens: int = 30000
    enable_rerank: bool = False
    user_prompt: str = ""
    stream: bool = False
    rewrite_enabled: bool = True
    mqe_enabled: bool = True
    mqe_variants: int = 3
    retrieval_fusion: str = "rrf"
    chunk_strategy_override: str | None = None


@dataclass
class QueryResult:
    """Structured retrieval result returned by ``EasyRAG.aquery``."""

    mode: QueryMode
    chunks: list[Document] = field(default_factory=list)
    citations: list[dict[str, str]] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseStorage(ABC):
    """Shared lifecycle methods for storage backends."""

    def __init__(self, working_dir: str, workspace: str) -> None:
        self.working_dir = working_dir
        self.workspace = workspace

    @abstractmethod
    async def initialize(self) -> None:
        """Load or create the backing store."""

    @abstractmethod
    async def finalize(self) -> None:
        """Persist in-memory state to disk."""


class BaseKVStorage(BaseStorage):
    """Store documents, chunks, summaries, and optional cache entries."""

    @abstractmethod
    async def upsert_documents(self, items: list[dict[str, Any]]) -> None:
        """Insert or update document records."""

    @abstractmethod
    async def upsert_chunks(self, items: list[dict[str, Any]]) -> None:
        """Insert or update chunk records."""

    @abstractmethod
    async def upsert_summaries(self, items: list[dict[str, Any]]) -> None:
        """Insert or update document summary records."""

    @abstractmethod
    async def upsert_cache(self, namespace: str, key: str, value: Any) -> None:
        """Persist a cache entry."""

    @abstractmethod
    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        """Return one document record if it exists."""

    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Return one chunk record if it exists."""

    @abstractmethod
    async def get_chunks(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """Return chunk records in caller-requested order where possible."""

    @abstractmethod
    async def get_summary(self, summary_id: str) -> dict[str, Any] | None:
        """Return one summary record if it exists."""

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """Return storage-specific counts."""


class BaseVectorStorage(BaseStorage):
    """Store chunk, entity, relation, and summary retrieval items."""

    @abstractmethod
    async def upsert(self, namespace: str, items: list[dict[str, Any]]) -> None:
        """Insert or update vector-like records in a namespace."""

    @abstractmethod
    async def similarity_search(self, namespace: str, query: str, top_k: int) -> list[dict[str, Any]]:
        """Search one namespace and return ranked records."""

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """Return namespace counts."""

    def set_embedding_func(self, embedding_func: Callable[[list[str]], list[list[float]]] | None) -> None:
        """Inject an embedding function into the storage backend."""

        del embedding_func

    def get_backend_name(self) -> str:
        """Return the active retrieval backend name."""

        return self.__class__.__name__


class BaseGraphStorage(BaseStorage):
    """Store graph nodes and edges for entity-aware retrieval."""

    @abstractmethod
    async def upsert_nodes(self, nodes: list[dict[str, Any]]) -> None:
        """Insert or update graph nodes."""

    @abstractmethod
    async def upsert_edges(self, edges: list[dict[str, Any]]) -> None:
        """Insert or update graph edges."""

    @abstractmethod
    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Return one graph node if it exists."""

    @abstractmethod
    async def get_neighbors(
        self,
        node_ids: list[str],
        *,
        kind_filter: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return ranked neighbor nodes."""

    @abstractmethod
    async def top_nodes(self, *, kind: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Return graph nodes ranked by centrality."""

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """Return node and edge counts."""


class BaseDocStatusStorage(BaseStorage):
    """Track document insertion lifecycle and errors."""

    @abstractmethod
    async def mark_status(
        self,
        document_id: str,
        status: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update document status."""

    @abstractmethod
    async def get_status(self, document_id: str) -> dict[str, Any] | None:
        """Return one document status if it exists."""

    @abstractmethod
    async def list_statuses(self) -> list[dict[str, Any]]:
        """Return all document statuses."""

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """Return status counts."""


RerankerFunc = Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]]
LLMFunc = Callable[..., Any]
EmbeddingFunc = Callable[[list[str]], list[list[float]]]
QueryModelFunc = Callable[..., str | list[str]]
ChunkerFunc = Callable[[Document], list[Document]]
