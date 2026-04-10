"""Integration-specific configuration helpers."""

from __future__ import annotations

import os


def get_github_mcp_command() -> str | None:
    """Return the configured GitHub MCP command, if any."""

    return os.getenv("REPOPILOT_GITHUB_MCP_COMMAND")


def get_github_mcp_args() -> str | None:
    """Return raw GitHub MCP args configuration."""

    return os.getenv("REPOPILOT_GITHUB_MCP_ARGS")


def get_github_mcp_env() -> str | None:
    """Return raw GitHub MCP env mapping configuration."""

    return os.getenv("REPOPILOT_GITHUB_MCP_ENV")


def get_github_mcp_url() -> str | None:
    """Return the configured GitHub MCP HTTP endpoint, if any."""

    return os.getenv("REPOPILOT_GITHUB_MCP_URL")


def get_github_mcp_headers() -> str | None:
    """Return raw GitHub MCP headers configuration."""

    return os.getenv("REPOPILOT_GITHUB_MCP_HEADERS")
