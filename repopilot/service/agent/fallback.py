"""Deterministic fallback agent for minimal environments."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from repopilot.service.schemas import SourceItem


class FallbackAgent:
    """Small deterministic agent used when LangChain dependencies are unavailable."""

    def __init__(self, tools: list[Any], system_prompt: str, rag_tool_getter: Callable[[], Any]) -> None:
        self.tools = tools
        self.system_prompt = system_prompt
        self._rag_tool_getter = rag_tool_getter

    def invoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Generate a bounded response from memory context and retrieved docs."""

        del config
        query = payload.get("user_query") or payload.get("messages", [{}])[-1].get("content", "")
        used_memory = list(payload.get("memory_context", []))
        doc_results = json.loads(self._rag_tool_getter().invoke({"query": query}))
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

    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Async compatibility wrapper so orchestration can prefer ainvoke."""

        return await asyncio.to_thread(self.invoke, payload, config)
