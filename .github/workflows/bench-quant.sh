#!/usr/bin/env bash
set -euo pipefail

NAME=$1
shift
MODEL="qwen25-$(echo "$NAME" | tr '[:upper:]_' '[:lower:]-')"

ollama create "$MODEL" -f /tmp/mf "$@"

# Save GGUF size to workspace so the TypeScript finalize step can read it
{
  manifest="${OLLAMA_MODELS}/manifests/registry.ollama.ai/library/${MODEL}/latest"
  if [[ -f "$manifest" ]]; then
    b=$(jq '[.layers[] | select(.mediaType | contains("model")) | .size] | add // 0' "$manifest")
    if (( b >= 1073741824 )); then
      awk "BEGIN{printf \"%.2f GiB\", $b/1073741824}"
    elif (( b >= 1048576 )); then
      awk "BEGIN{printf \"%.2f MiB\", $b/1048576}"
    else
      echo "${b} B"
    fi
  fi
} > "${GITHUB_WORKSPACE}/.bench-sizes/${NAME}" 2>/dev/null || true
