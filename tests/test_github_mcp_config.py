"""Tests for RepoPilot-side GitHub MCP configuration parsing."""

from __future__ import annotations

import os
import unittest

from repopilot.service.integrations.github_mcp import build_github_server_config


class GitHubMCPConfigTestCase(unittest.TestCase):
    """Verify that `.env`-style settings become the expected MCP config shape."""

    def setUp(self) -> None:
        self.names = [
            "REPOPILOT_GITHUB_MCP_COMMAND",
            "REPOPILOT_GITHUB_MCP_ARGS",
            "REPOPILOT_GITHUB_MCP_ENV",
            "REPOPILOT_GITHUB_MCP_URL",
            "REPOPILOT_GITHUB_MCP_HEADERS",
        ]
        self.old_values = {name: os.environ.get(name) for name in self.names}
        for name in self.names:
            os.environ.pop(name, None)

    def tearDown(self) -> None:
        for name, value in self.old_values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_stdio_config_supports_args_and_env(self) -> None:
        os.environ["REPOPILOT_GITHUB_MCP_COMMAND"] = "docker"
        os.environ["REPOPILOT_GITHUB_MCP_ARGS"] = '["run", "-i", "--rm", "ghcr.io/github/github-mcp-server", "stdio"]'
        os.environ["REPOPILOT_GITHUB_MCP_ENV"] = '{"GITHUB_PERSONAL_ACCESS_TOKEN":"abc"}'

        config = build_github_server_config()

        self.assertEqual(config["github"]["transport"], "stdio")
        self.assertEqual(config["github"]["command"], "docker")
        self.assertEqual(config["github"]["args"][-1], "stdio")
        self.assertEqual(config["github"]["env"]["GITHUB_PERSONAL_ACCESS_TOKEN"], "abc")

    def test_http_config_supports_headers(self) -> None:
        os.environ["REPOPILOT_GITHUB_MCP_URL"] = "http://localhost:8000/mcp"
        os.environ["REPOPILOT_GITHUB_MCP_HEADERS"] = '{"Authorization":"Bearer test"}'

        config = build_github_server_config()

        self.assertEqual(config["github"]["transport"], "streamable_http")
        self.assertEqual(config["github"]["url"], "http://localhost:8000/mcp")
        self.assertEqual(config["github"]["headers"]["Authorization"], "Bearer test")


if __name__ == "__main__":
    unittest.main()
