"""Project configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional at import time.
    load_dotenv = None


def _discover_project_root() -> Path:
    """Best-effort project root discovery used before env vars are loaded."""

    return Path(__file__).resolve().parent.parent


def load_environment() -> Path:
    """
    Load the project's `.env` file once and return the effective project root.

    Why:
        RepoPilot has a growing set of runtime knobs for model access, local
        storage, and GitHub MCP wiring. Loading `.env` centrally keeps those
        settings discoverable and avoids scattering `load_dotenv()` calls across
        unrelated modules.

    Behavior:
        - Look for `.env` in the repository root beside `streamlit_app.py`
        - Respect already-exported environment variables
        - Succeed silently when `python-dotenv` or the file is absent
    """

    project_root = _discover_project_root()
    env_path = project_root / ".env"
    if load_dotenv is not None and env_path.exists():
        load_dotenv(env_path, override=False)
    return project_root


PROJECT_ROOT = load_environment()


def get_repo_root() -> Path:
    """
    Return the repository root used for document discovery and local storage.

    Preference order:
        1. `REPOPILOT_REPO_ROOT` from the environment or `.env`
        2. The discovered repository root containing this codebase
    """

    return Path(os.getenv("REPOPILOT_REPO_ROOT", PROJECT_ROOT)).resolve()


def get_data_dir() -> Path:
    """
    Return the writable local data directory for indexes and fallback memory.

    This directory stays inside the repository by default so the demo remains
    self-contained and does not require hidden directories elsewhere on disk.
    """

    return Path(os.getenv("REPOPILOT_DATA_DIR", get_repo_root() / ".repopilot")).resolve()


def get_rag_index_path() -> Path:
    """Return the on-disk location for the lightweight document index."""

    return Path(os.getenv("REPOPILOT_RAG_INDEX_PATH", get_data_dir() / "rag_index.json")).resolve()


def get_memory_store_path() -> Path:
    """Return the fallback JSON file used when mem0 is unavailable."""

    return Path(os.getenv("REPOPILOT_MEMORY_STORE_PATH", get_data_dir() / "memory_store.json")).resolve()


def get_default_model() -> str:
    """
    Return the chat model identifier used when LangChain is available.

    Example values:
        - `openai:gpt-4.1-mini`
        - `openai:gpt-4.1`
    """

    return os.getenv("REPOPILOT_MODEL", "openai:gpt-4.1-mini")


def get_model_name() -> str:
    """
    Return the raw model name sent to an OpenAI-compatible chat endpoint.

    Examples:
        - `gpt-4.1-mini`
        - `deepseek-chat`
    """

    return os.getenv("REPOPILOT_MODEL_NAME", get_default_model().split(":", 1)[-1])


def get_openai_api_key() -> str:
    """
    Return the API key used by the OpenAI-compatible chat client.

    This keeps the project compatible with providers such as OpenAI and
    DeepSeek, both of which can expose OpenAI-style HTTP interfaces.
    """

    return os.getenv("OPENAI_API_KEY", "")


def get_openai_base_url() -> str | None:
    """
    Return the optional OpenAI-compatible base URL.

    Use this to point RepoPilot at providers such as DeepSeek without changing
    the agent code. If unset, LangChain falls back to the default OpenAI API.
    """

    value = os.getenv("OPENAI_BASE_URL", "").strip()
    return value or None
