"""Smoke-test RepoPilot's GitHub MCP configuration."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repopilot.integrations.github_mcp import build_github_server_config, load_github_tools


async def _main() -> None:
    """
    Print the effective GitHub MCP configuration and the discovered read-only tools.

    Why:
        RepoPilot's MCP wiring depends on runtime `.env` values, so a dedicated
        smoke test is faster than debugging full agent startup when MCP is not
        connected yet.
    """

    config = build_github_server_config()
    print("server_config=")
    print(json.dumps(config, ensure_ascii=False, indent=2))

    tools = await load_github_tools()
    print(f"tool_count={len(tools)}")
    for tool in tools:
        print(getattr(tool, "name", type(tool).__name__))


if __name__ == "__main__":
    asyncio.run(_main())
