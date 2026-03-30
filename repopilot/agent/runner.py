"""Main agent orchestration for RepoPilot."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from repopilot.compat import Runnable
from repopilot.config import get_model_name, get_openai_api_key, get_openai_base_url
from repopilot.integrations.github_mcp import load_github_tools
from repopilot.rag.tool import search_docs_tool
from repopilot.response_builder import build_final_response
from repopilot.schemas import FinalResponse, SourceItem

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


class FallbackAgent:
    """Small deterministic agent used when LangChain dependencies are unavailable."""

    def __init__(self, tools: list[Any], system_prompt: str) -> None:
        self.tools = tools
        self.system_prompt = system_prompt

    def invoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Generate a bounded response from memory context and retrieved docs."""

        del config  # The fallback path does not use configurable runtime today.
        query = payload.get("user_query") or payload.get("messages", [{}])[-1].get("content", "")
        used_memory = list(payload.get("memory_context", []))
        doc_results = json.loads(search_docs_tool.invoke({"query": query}))
        citations = [SourceItem.model_validate(item) for item in doc_results]

        answer_lines = []
        if used_memory:
            answer_lines.append("I considered your saved preferences or ongoing task context when preparing this answer.")
        if citations:
            answer_lines.append("I found relevant internal documentation that directly addresses your question.")
            answer_lines.extend(f"- {item.title}: {item.snippet[:180].strip()}" for item in citations[:3])
            confidence = "high" if len(citations) >= 2 else "medium"
        else:
            if any(token in query.lower() for token in ("issue", "pr", "pull request", "github", "file")):
                answer_lines.append(
                    "I could not access GitHub MCP evidence in the current environment, so I cannot give a grounded repository answer yet."
                )
            else:
                answer_lines.append("I could not find supporting documentation in the local RAG index for this question.")
            confidence = "low"

        return {
            "answer": "\n".join(answer_lines),
            "citations": citations,
            "used_memory": used_memory,
            "confidence": confidence,
        }


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


def build_system_prompt() -> str:
    """
    Define the assistant's operating rules.

    Must enforce:
        - answer only within repo-assistant scope
        - prefer retrieved evidence over guesswork
        - include citations whenever evidence is available
        - say when evidence is insufficient
    """

    return (
        "You are RepoPilot, a single-repository engineering knowledge assistant. "
        "Use documentation retrieval and approved GitHub read-only tools to answer questions. "
        "Never claim certainty without evidence. Prefer concise answers with explicit source grounding. "
        "If evidence is missing or a tool is unavailable, say so clearly."
    )


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

    tools = [search_docs_tool, *_load_github_tools_sync()]
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

    _CACHED_AGENT = FallbackAgent(tools=tools, system_prompt=system_prompt)
    return _CACHED_AGENT


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
        raw_result = agent.invoke(payload, config={"configurable": {"thread_id": thread_id, "user_id": user_id}})
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
