"""GitHub MCP integration for the service layer."""

from __future__ import annotations

import json
import shlex
from typing import Any

from repopilot.config import (
    get_github_mcp_args,
    get_github_mcp_command,
    get_github_mcp_env,
    get_github_mcp_headers,
    get_github_mcp_url,
    load_environment,
)
from repopilot.service.schemas import SourceItem

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:  # pragma: no cover - exercised only without optional deps.
    MultiServerMCPClient = None

try:
    from langchain_core.tools import BaseTool
except ImportError:  # pragma: no cover - exercised only without optional deps.
    from repopilot.support.optional_deps import BaseTool

_WRITE_MARKERS = ("create", "update", "delete", "merge", "comment", "review", "push", "write")

load_environment()


def _parse_json_mapping(value: str | None) -> dict[str, str]:
    """
    Parse an optional JSON object string used for MCP env vars or headers.

    Why:
        `.env` files only store strings, but MCP configuration often needs
        structured key/value maps. A narrow JSON parser keeps the public config
        surface explicit and avoids custom mini-languages.
    """

    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): str(val) for key, val in parsed.items()}


def _parse_command_args(value: str | None) -> list[str]:
    """
    Parse optional stdio command arguments from `.env`.

    Supported formats:
        - shell-like string: `stdio --read-only`
        - JSON array: `["stdio", "--read-only"]`
    """

    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return shlex.split(value)


def build_github_server_config() -> dict[str, dict[str, Any]]:
    """
    Build the RepoPilot-side MCP server configuration for GitHub access.

    Supported `.env` inputs:
        - `REPOPILOT_GITHUB_MCP_COMMAND`
        - `REPOPILOT_GITHUB_MCP_ARGS`
        - `REPOPILOT_GITHUB_MCP_ENV`
        - `REPOPILOT_GITHUB_MCP_URL`
        - `REPOPILOT_GITHUB_MCP_HEADERS`

    Transport behavior:
        - If URL is set, use `streamable_http`
        - Otherwise, if COMMAND is set, use `stdio`
        - If neither is set, return an empty config
    """

    command = get_github_mcp_command()
    url = get_github_mcp_url()
    args = _parse_command_args(get_github_mcp_args())
    env = _parse_json_mapping(get_github_mcp_env())
    headers = _parse_json_mapping(get_github_mcp_headers())

    if url:
        config: dict[str, Any] = {"transport": "streamable_http", "url": url}
        if headers:
            config["headers"] = headers
        return {"github": config}

    if command:
        config = {"transport": "stdio", "command": command, "args": args}
        if env:
            config["env"] = env
        return {"github": config}

    return {}


async def load_github_tools() -> list[BaseTool]:
    """
    Load the approved GitHub MCP tools for this assistant.

    Allowed capability examples:
        - fetch file content
        - inspect issue / PR metadata
        - search code within the repo

    Excluded:
        - write operations
        - PR creation
        - code modification flows

    Failure strategy:
        Return an empty list when the adapter is not installed or the MCP server
        is not configured yet, so the rest of the app can still run in a local
        demo mode focused on documentation RAG.
    """

    if MultiServerMCPClient is None:
        return []

    server_config = build_github_server_config()
    if not server_config:
        return []

    try:
        client = MultiServerMCPClient(server_config)
        tools = await client.get_tools()
    except Exception:
        return []

    approved: list[BaseTool] = []
    for tool in tools:
        name = getattr(tool, "name", "").lower()
        if any(marker in name for marker in _WRITE_MARKERS):
            continue
        approved.append(tool)
    return approved


def normalize_github_result(raw_result: Any) -> SourceItem:
    """
    Convert GitHub MCP output into a shared internal source format.

    Why:
        RAG citations and GitHub citations should flow through the same response
        builder so the UI never needs source-specific rendering branches.
    """

    if isinstance(raw_result, SourceItem):
        return raw_result

    if isinstance(raw_result, dict):
        return SourceItem(
            source_type="github",
            title=str(raw_result.get("title") or raw_result.get("name") or "GitHub result"),
            location=str(raw_result.get("url") or raw_result.get("path") or ""),
            snippet=str(raw_result.get("snippet") or raw_result.get("body") or raw_result.get("content") or "")[:400],
        )

    return SourceItem(
        source_type="github",
        title="GitHub result",
        location="",
        snippet=str(raw_result)[:400],
    )
