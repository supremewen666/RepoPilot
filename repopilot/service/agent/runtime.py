"""Main service-agent orchestration for RepoPilot."""

from __future__ import annotations

import asyncio
from typing import Any

from repopilot.support.optional_deps import Runnable
from repopilot.config import get_model_name, get_openai_api_key, get_openai_base_url
from repopilot.service.agent.fallback import FallbackAgent
from repopilot.service.agent.prompts import build_system_prompt
from repopilot.service.agent.tools import create_search_docs_tool
from repopilot.service.integrations.github_mcp import load_github_tools
from repopilot.service.response_builder import build_final_response
from repopilot.service.schemas import FinalResponse
from repopilot.rag import EasyRAG

try:
    from langchain.agents import create_agent
except ImportError:  # pragma: no cover - exercised only without optional deps.
    create_agent = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - exercised only without optional deps.
    ChatOpenAI = None

_THREAD_HISTORY: dict[str, list[str]] = {}
_CACHED_AGENT: Runnable | None = None
_CACHED_RAG: EasyRAG | None = None
_CACHED_RAG_TOOL: Any | None = None


def _load_github_tools_sync() -> list[Any]:
    """Bridge the async MCP loader into the synchronous app startup path."""

    try:
        return asyncio.run(load_github_tools())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(load_github_tools())
        finally:
            loop.close()


def _run_async(awaitable: object) -> object:
    """Run async RAG operations from the synchronous orchestration layer."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(awaitable)
    finally:
        loop.close()


def _get_rag() -> EasyRAG:
    """Return the singleton EasyRAG instance used by the agent."""

    global _CACHED_RAG
    if _CACHED_RAG is not None:
        return _CACHED_RAG
    rag = EasyRAG()
    _run_async(rag.initialize_storages())
    _CACHED_RAG = rag
    return rag


def _get_rag_tool() -> Any:
    """Return the model-visible RAG tool bound to the singleton EasyRAG instance."""

    global _CACHED_RAG_TOOL
    if _CACHED_RAG_TOOL is not None:
        return _CACHED_RAG_TOOL
    rag = _get_rag()
    default_mode = "mix" if rag.can_rerank() else "hybrid"
    _CACHED_RAG_TOOL = create_search_docs_tool(
        lambda: rag,
        default_mode=default_mode,
        rewrite_enabled=True,
        mqe_enabled=True,
    )
    return _CACHED_RAG_TOOL


def build_agent() -> Runnable:
    """
    Build the single production agent for RepoPilot.

    Responsibilities:
        - register RAG and GitHub MCP tools
        - inject short-term memory / thread awareness
        - enforce structured final output
        - keep the agent limited to this repository assistant scope

    Non-goals:
        - no multi-agent delegation
        - no code modification actions
        - no external MCP servers beyond GitHub
    """

    global _CACHED_AGENT
    if _CACHED_AGENT is not None:
        return _CACHED_AGENT

    tools = [_get_rag_tool(), *_load_github_tools_sync()]
    system_prompt = build_system_prompt()
    if create_agent is not None and ChatOpenAI is not None:
        try:
            model = ChatOpenAI(
                model=get_model_name(),
                api_key=get_openai_api_key(),
                base_url=get_openai_base_url(),
            )
            _CACHED_AGENT = create_agent(
                model=model,
                tools=tools,
                system_prompt=system_prompt,
                response_format=FinalResponse,
            )
            return _CACHED_AGENT
        except Exception:
            pass

    _CACHED_AGENT = FallbackAgent(tools=tools, system_prompt=system_prompt, rag_tool_getter=_get_rag_tool)
    return _CACHED_AGENT


def _build_docs_only_fallback() -> FallbackAgent:
    """Return a local-only fallback agent that avoids GitHub MCP tools."""

    return FallbackAgent([_get_rag_tool()], build_system_prompt(), _get_rag_tool)


def _is_github_repo_resolution_error(exc: Exception) -> bool:
    """Return whether the exception looks like a GitHub MCP repository-resolution failure."""

    message = str(exc).lower()
    markers = (
        "failed to resolve git reference",
        "failed to get repository info",
        "api.github.com/repos",
        "404 not found",
    )
    return all(marker in message for marker in markers[:2]) or (
        "api.github.com/repos" in message and "404 not found" in message
    )


def _fallback_to_docs_only(user_query: str, memory_context: list[str]) -> FinalResponse:
    """Answer from local docs only when GitHub MCP is unavailable or misconfigured."""

    fallback = _build_docs_only_fallback()
    result = fallback.invoke(
        {
            "messages": [{"role": "user", "content": user_query}],
            "user_query": user_query,
            "memory_context": memory_context,
        }
    )
    answer = str(result.get("answer", "")).strip()
    note = "GitHub evidence was unavailable, so I answered from local documentation only."
    if answer:
        result["answer"] = f"{note}\n\n{answer}"
    else:
        result["answer"] = note
    return build_final_response(result)


def _invoke_agent_runnable(agent: Runnable, payload: dict[str, Any], config: dict[str, Any]) -> Any:
    """Execute the runnable while preferring async paths for async-only tools."""

    ainvoke = getattr(agent, "ainvoke", None)
    if callable(ainvoke):
        return _run_async(ainvoke(payload, config=config))

    invoke = getattr(agent, "invoke", None)
    if callable(invoke):
        return invoke(payload, config=config)

    raise RuntimeError("Agent does not expose invoke or ainvoke.")


def invoke_agent(user_query: str, user_id: str, thread_id: str, memory_context: list[str]) -> FinalResponse:
    """
    Execute one agent turn with all required runtime context.

    Inputs:
        - user_query: current user question
        - user_id: stable identity for long-term memory lookup
        - thread_id: conversation identity for short-term memory
        - memory_context: relevant user/task memories selected before invocation

    Returns:
        Structured FinalResponse object ready for UI rendering.

    Failure strategy:
        If one tool path fails, the function still returns a bounded answer that
        explains what evidence source was missing instead of raising into the UI.
    """

    history = _THREAD_HISTORY.setdefault(thread_id, [])
    history.append(user_query)
    agent = build_agent()
    payload = {
        "messages": [{"role": "user", "content": user_query}],
        "user_query": user_query,
        "user_id": user_id,
        "memory_context": memory_context,
        "history": history[-10:],
    }

    try:
        raw_result = _invoke_agent_runnable(
            agent,
            payload,
            config={"configurable": {"thread_id": thread_id, "user_id": user_id}},
        )
        if hasattr(raw_result, "model_dump"):
            return build_final_response(raw_result.model_dump())
        if isinstance(raw_result, dict):
            structured_response = raw_result.get("structured_response")
            if structured_response is not None:
                if hasattr(structured_response, "model_dump"):
                    return build_final_response(structured_response.model_dump())
                if isinstance(structured_response, dict):
                    return build_final_response(structured_response)
            return build_final_response(raw_result)
    except Exception as exc:
        if _is_github_repo_resolution_error(exc):
            try:
                return _fallback_to_docs_only(user_query, memory_context)
            except Exception:
                pass
        return build_final_response(
            {
                "answer": f"I ran into a tool-orchestration error and could not finish the grounded answer: {exc}",
                "citations": [],
                "used_memory": memory_context,
                "confidence": "low",
            }
        )

    return build_final_response(
        {
            "answer": "I could not interpret the agent output into the expected response schema.",
            "citations": [],
            "used_memory": memory_context,
            "confidence": "low",
        }
    )
