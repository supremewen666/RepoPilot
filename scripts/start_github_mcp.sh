#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_BIN="${PROJECT_ROOT}/bin/github-mcp-server"

if [[ ! -x "${SERVER_BIN}" ]]; then
  echo "Missing executable GitHub MCP server at ${SERVER_BIN}" >&2
  echo "Place the official github-mcp-server binary there, then retry." >&2
  exit 1
fi

if [[ -z "${GITHUB_PERSONAL_ACCESS_TOKEN:-}" ]]; then
  echo "GITHUB_PERSONAL_ACCESS_TOKEN is not set." >&2
  exit 1
fi

exec "${SERVER_BIN}" stdio
