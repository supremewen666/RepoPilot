"""FastAPI application exposing RepoPilot's RAG and KG interfaces."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable

from fastapi import FastAPI, HTTPException

from repopilot.config import get_rag_working_dir, get_rag_workspace
from repopilot.rag import EasyRAG, QueryParam
from repopilot.service.api.schemas import (
    CustomKGRequest,
    EntityMergeRequest,
    EntityPayload,
    EntityUpdatePayload,
    IndexTaskRequest,
    QueryRequest,
    QueryResponse,
    QueryResponseChunk,
    RelationMergeRequest,
    RelationPayload,
    RelationUpdatePayload,
)
from repopilot.service.tasks.manager import IndexTaskManager


@asynccontextmanager
async def _open_rag(app: FastAPI, workspace: str | None) -> AsyncIterator[EasyRAG]:
    """Open one EasyRAG instance for a single request."""

    factory = getattr(app.state, "rag_factory", None)
    rag = factory(workspace) if callable(factory) else EasyRAG(working_dir=get_rag_working_dir(), workspace=workspace or get_rag_workspace())
    await rag.initialize_storages()
    try:
        yield rag
    finally:
        await rag.finalize_storages()


def _serialize_query_result(result: object) -> QueryResponse:
    """Convert one EasyRAG QueryResult into the REST response model."""

    return QueryResponse(
        mode=str(getattr(result, "mode", "")),
        chunks=[
            QueryResponseChunk(page_content=document.page_content, metadata=dict(document.metadata))
            for document in list(getattr(result, "chunks", []))
        ],
        citations=list(getattr(result, "citations", [])),
        entities=list(getattr(result, "entities", [])),
        relations=list(getattr(result, "relations", [])),
        metadata=dict(getattr(result, "metadata", {})),
    )


def create_app(
    *,
    rag_factory: Callable[[str | None], EasyRAG] | None = None,
    task_manager: IndexTaskManager | None = None,
) -> FastAPI:
    """Create the RepoPilot RAG service application."""

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        manager = task_manager or IndexTaskManager(working_dir=get_rag_working_dir(), default_workspace=get_rag_workspace())
        await manager.initialize()
        app.state.index_task_manager = manager
        app.state.rag_factory = rag_factory
        try:
            yield
        finally:
            await manager.finalize()

    app = FastAPI(title="RepoPilot RAG API", version="0.1.0", lifespan=_lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/rag/query", response_model=QueryResponse)
    async def query_rag(request: QueryRequest) -> QueryResponse:
        async with _open_rag(app, request.workspace) as rag:
            result = await rag.aquery(
                request.query,
                QueryParam(
                    mode=request.mode,
                    top_k=request.top_k,
                    chunk_top_k=request.chunk_top_k,
                    enable_rerank=request.enable_rerank,
                ),
            )
            return _serialize_query_result(result)

    @app.post("/rag/index/tasks", status_code=202)
    async def create_index_task(request: IndexTaskRequest) -> dict[str, object]:
        try:
            task = await app.state.index_task_manager.create_task(
                action=request.action,
                workspace=request.workspace,
                doc_ids=request.doc_ids,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"task_id": task["task_id"], "status": task["status"]}

    @app.get("/rag/index/tasks")
    async def list_index_tasks(limit: int = 100) -> list[dict[str, object]]:
        return await app.state.index_task_manager.list_tasks(limit=limit)

    @app.get("/rag/index/tasks/{task_id}")
    async def get_index_task(task_id: str) -> dict[str, object]:
        task = await app.state.index_task_manager.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
        return task

    @app.post("/rag/kg/entities")
    async def create_entity(request: EntityPayload) -> dict[str, object]:
        async with _open_rag(app, None) as rag:
            return await rag.acreate_entity(
                entity_id=request.id,
                label=request.label,
                entity_types=request.entity_types,
                description=request.description,
                aliases=request.aliases,
                metadata=request.metadata,
                provenance=request.provenance,
            )

    @app.patch("/rag/kg/entities/{entity_id}")
    async def update_entity(entity_id: str, request: EntityUpdatePayload) -> dict[str, object]:
        async with _open_rag(app, None) as rag:
            try:
                return await rag.aupdate_entity(
                    entity_id,
                    label=request.label,
                    entity_types=request.entity_types,
                    description=request.description,
                    aliases=request.aliases,
                    metadata=request.metadata,
                    provenance=request.provenance,
                )
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/rag/kg/entities/{entity_id}")
    async def delete_entity(entity_id: str) -> dict[str, object]:
        async with _open_rag(app, None) as rag:
            return await rag.adelete_entity(entity_id)

    @app.post("/rag/kg/entities/merge")
    async def merge_entities(request: EntityMergeRequest) -> dict[str, object]:
        async with _open_rag(app, None) as rag:
            return await rag.amerge_entities(request.source_entity_id, request.target_entity_id)

    @app.post("/rag/kg/relations")
    async def create_relation(request: RelationPayload) -> dict[str, object]:
        async with _open_rag(app, None) as rag:
            try:
                return await rag.acreate_relation(
                    relation_id=request.id,
                    source_entity_id=request.source_entity_id,
                    target_entity_id=request.target_entity_id,
                    relation=request.relation,
                    description=request.description,
                    weight=request.weight,
                    metadata=request.metadata,
                    provenance=request.provenance,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.patch("/rag/kg/relations/{relation_id}")
    async def update_relation(relation_id: str, request: RelationUpdatePayload) -> dict[str, object]:
        async with _open_rag(app, None) as rag:
            try:
                return await rag.aupdate_relation(
                    relation_id,
                    source_entity_id=request.source_entity_id,
                    target_entity_id=request.target_entity_id,
                    relation=request.relation,
                    description=request.description,
                    weight=request.weight,
                    metadata=request.metadata,
                    provenance=request.provenance,
                )
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/rag/kg/relations/{relation_id}")
    async def delete_relation(relation_id: str) -> dict[str, object]:
        async with _open_rag(app, None) as rag:
            return await rag.adelete_relation(relation_id)

    @app.post("/rag/kg/relations/merge")
    async def merge_relations(request: RelationMergeRequest) -> dict[str, object]:
        async with _open_rag(app, None) as rag:
            return await rag.amerge_relations(request.source_relation_id, request.target_relation_id)

    @app.post("/rag/kg/custom")
    async def insert_custom_kg(request: CustomKGRequest) -> dict[str, object]:
        async with _open_rag(app, None) as rag:
            return await rag.ainsert_custom_kg(
                entities=[item.model_dump() for item in request.entities],
                relations=[item.model_dump() for item in request.relations],
                source_label=request.source_label,
                batch_id=request.batch_id,
            )

    return app
