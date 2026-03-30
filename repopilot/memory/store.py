"""Long-term memory support with mem0-first behavior and a local JSON fallback."""

from __future__ import annotations

import json
from pathlib import Path

from repopilot.config import get_memory_store_path

try:
    from mem0 import MemoryClient  # type: ignore
except Exception:  # pragma: no cover - optional dependency may fail during import/setup.
    MemoryClient = None


def _load_fallback_store() -> dict[str, list[str]]:
    """Read the local JSON memory store used when mem0 is not available."""

    path = get_memory_store_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_fallback_store(payload: dict[str, list[str]]) -> None:
    """Persist the local JSON memory store in a repo-local writable directory."""

    path = get_memory_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_memory_client() -> MemoryClient | None:
    """Create a mem0 client when the dependency and credentials are present."""

    if MemoryClient is None:
        return None
    try:
        return MemoryClient()
    except Exception:
        return None


def get_relevant_memories(user_id: str, query: str) -> list[str]:
    """
    Fetch only the memories that may help answer the current question.

    Inputs:
        - user_id: stable identifier for one user across sessions
        - query: current question used to filter semantically relevant memory

    Returns:
        A small list of durable memory strings. These are intended to include
        preferences and active task context, not arbitrary chat history.

    Failure strategy:
        If mem0 is unavailable or errors, fall back to the local JSON store and
        return at most the newest few items to avoid prompt pollution.
    """

    client = _build_memory_client()
    if client is not None:
        try:
            results = client.search(query=query, user_id=user_id, limit=5)
            memories = []
            for item in results or []:
                text = item.get("memory") if isinstance(item, dict) else str(item)
                if text:
                    memories.append(text)
            return memories[:5]
        except Exception:
            pass

    fallback = _load_fallback_store()
    return fallback.get(user_id, [])[-5:]


def should_persist_memory(user_query: str, assistant_answer: str) -> bool:
    """
    Decide whether the current turn contains durable information worth saving.

    Save:
        - stable preferences such as language or response-style choices
        - ongoing task context likely to matter in later turns

    Do not save:
        - transient retrieval snippets
        - one-off factual questions without future value

    Heuristic note:
        A narrow heuristic is preferable here because over-saving memory harms
        answer quality more than under-saving in a small assistant.
    """

    text = f"{user_query}\n{assistant_answer}".lower()
    preference_markers = ("prefer", "i like", "please answer", "use chinese", "中文", "偏好", "喜欢")
    task_markers = ("working on", "currently", "task", "issue", "pr", "bug", "investigating", "正在")
    return any(marker in text for marker in preference_markers + task_markers)


def _summarize_memory(user_query: str) -> str:
    """Extract a compact memory candidate from the user query."""

    trimmed = " ".join(user_query.strip().split())
    return trimmed[:240]


def save_memory_if_needed(user_id: str, user_query: str, assistant_answer: str) -> None:
    """
    Persist useful long-term memory under a narrow policy.

    Inputs:
        - user_id: stable identifier used as the memory namespace
        - user_query: current user utterance
        - assistant_answer: latest answer, included so persistence decisions can
          account for the full turn instead of raw user text alone

    Failure strategy:
        Attempt mem0 first. If mem0 is missing or fails, write to a local JSON
        file so the project remains demoable in a minimal environment.
    """

    if not should_persist_memory(user_query=user_query, assistant_answer=assistant_answer):
        return

    memory_text = _summarize_memory(user_query)
    client = _build_memory_client()
    if client is not None:
        try:
            client.add(messages=[{"role": "user", "content": memory_text}], user_id=user_id)
            return
        except Exception:
            pass

    fallback = _load_fallback_store()
    bucket = fallback.setdefault(user_id, [])
    if memory_text not in bucket:
        bucket.append(memory_text)
    _save_fallback_store(fallback)
