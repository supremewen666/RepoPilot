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

infer_github_repository() {
  local remote_url
  local sanitized_url
  remote_url="$(git -C "${PROJECT_ROOT}" remote get-url origin 2>/dev/null || true)"
  if [[ -z "${remote_url}" ]]; then
    return 0
  fi

  sanitized_url="${remote_url%.git}"
  if [[ "${sanitized_url}" =~ github\.com[:/]([^/]+)/([^/]+)$ ]]; then
    printf '%s/%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
  fi
}

cd "${PROJECT_ROOT}"

if [[ -z "${GITHUB_REPOSITORY:-}" ]]; then
  inferred_repository="$(infer_github_repository)"
  if [[ -n "${inferred_repository}" ]]; then
    export GITHUB_REPOSITORY="${inferred_repository}"
  fi
fi

exec "${SERVER_BIN}" stdio --read-only
