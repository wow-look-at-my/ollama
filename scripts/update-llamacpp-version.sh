#!/usr/bin/env bash
# Resolve the llama.cpp fork's master tip to a concrete commit SHA and write it
# into LLAMA_CPP_VERSION.
#
# Why this exists: the Dockerfile fetches llama.cpp via CMake FetchContent using
# the contents of LLAMA_CPP_VERSION as the git ref, and `COPY LLAMA_CPP_VERSION`
# is what keys the (expensive) llama.cpp clone+compile layer in Docker's cache.
# A moving branch name like "master" never changes that file's content, so Docker
# treats the layer as a permanent cache hit and silently ships a stale binary even
# after master moves. Pinning the *resolved SHA* fixes both directions:
#   - master moved  -> SHA changes -> file content changes -> cache busts -> rebuild
#   - master same   -> SHA identical -> file content identical -> cached layers reused
# Run this in CI after checkout and before any step that reads LLAMA_CPP_VERSION
# (the Docker build, and the enginehash/vendorsha cache-key steps).
set -euo pipefail

REPO="${LLAMA_CPP_REPO:-https://github.com/wow-look-at-my/llama.cpp.git}"
REF="${LLAMA_CPP_REF:-master}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version_file="${repo_root}/LLAMA_CPP_VERSION"

old="$(cat "${version_file}" 2>/dev/null | awk 'NR==1 {print $1}' || true)"

# A full 40-hex SHA in the checked-in file is a deliberate pin: keep it, so
# builds are insulated from pushes to the llama.cpp fork and the file content
# (the Docker COPY cache key) changes exactly when the pin is advanced. A
# branch name (e.g. "master") still resolves to its tip SHA below. To advance
# the pin, re-resolve explicitly: LLAMA_CPP_REF=master scripts/update-llamacpp-version.sh
if [ -z "${LLAMA_CPP_REF:-}" ] && printf '%s' "${old}" | grep -Eq '^[0-9a-f]{40}$'; then
  echo "LLAMA_CPP_VERSION pinned to ${old} — keeping the checked-in pin"
  exit 0
fi

# git ls-remote needs no clone, no API token, and is not rate-limited like the
# REST API. For a branch ref it prints exactly one "<sha>\t<ref>" line.
sha="$(git ls-remote "${REPO}" "refs/heads/${REF}" | awk 'NR==1 {print $1}')"

if ! printf '%s' "${sha}" | grep -Eq '^[0-9a-f]{40}$'; then
  echo "error: could not resolve ${REF} on ${REPO} (got: '${sha}')" >&2
  exit 1
fi

printf '%s\n' "${sha}" > "${version_file}"

if [ "${old}" = "${sha}" ]; then
  echo "LLAMA_CPP_VERSION unchanged: ${sha} (${REF} @ ${REPO}) — caches reusable"
else
  echo "LLAMA_CPP_VERSION ${old:-<none>} -> ${sha} (${REF} @ ${REPO}) — caches will bust"
fi
