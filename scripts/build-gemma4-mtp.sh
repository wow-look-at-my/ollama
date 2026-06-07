#!/usr/bin/env bash
#
# Build the Gemma 4 Multi-Token-Prediction model for this fork: the full Gemma 4
# target plus a SEPARATE gemma4_assistant drafter GGUF (attached at serve time via
# --mtp-head). One command: download the Hugging Face safetensors, write a
# Modelfile, and run `ollama create`.
#
# Prerequisites:
#   - this fork's `ollama` built and on PATH, with `ollama serve` on localhost
#   - HF_TOKEN with read access to the (gated) google/gemma-4 repos
#   - `hf` or `huggingface-cli` on PATH (pip install -U huggingface_hub)
#
# Usage:
#   HF_TOKEN=hf_xxx scripts/build-gemma4-mtp.sh
#   MODEL_NAME=gemma4-mtp QUANTIZE=q4_K_M scripts/build-gemma4-mtp.sh
#
# See docs/gemma4-mtp.md for details.
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-gemma4-mtp}"
TARGET_REPO="${TARGET_REPO:-google/gemma-4-31B-it}"
DRAFT_REPO="${DRAFT_REPO:-google/gemma-4-31B-it-assistant}"
QUANTIZE="${QUANTIZE:-q4_K_M}"
DRAFT_QUANTIZE="${DRAFT_QUANTIZE:-}"
WORKDIR="${WORKDIR:-${HOME}/gemma4-mtp}"

die() { echo "error: $*" >&2; exit 1; }

command -v ollama >/dev/null 2>&1 || \
  die "ollama not on PATH — build this fork first: go build -o /usr/bin/ollama ."
: "${HF_TOKEN:?set HF_TOKEN to a token with read access to ${TARGET_REPO}}"
export HF_TOKEN

# Skip pytorch/original duplicates and any prebuilt GGUFs; we only need
# config.json, *.safetensors and the tokenizer files.
excludes=(--exclude "original/*" "*.pth" "*.gguf" "consolidated.*")

if command -v hf >/dev/null 2>&1; then
  hf_dl() { hf download "$1" --local-dir "$2" "${excludes[@]}"; }
elif command -v huggingface-cli >/dev/null 2>&1; then
  hf_dl() { huggingface-cli download "$1" --local-dir "$2" "${excludes[@]}"; }
else
  die "need 'hf' or 'huggingface-cli' on PATH (pip install -U huggingface_hub)"
fi

target_dir="${WORKDIR}/target"
draft_dir="${WORKDIR}/draft"
mkdir -p "${target_dir}" "${draft_dir}"

echo ">> downloading target ${TARGET_REPO} -> ${target_dir} (large: ~60GB at bf16)"
hf_dl "${TARGET_REPO}" "${target_dir}"

echo ">> downloading drafter ${DRAFT_REPO} -> ${draft_dir}"
hf_dl "${DRAFT_REPO}" "${draft_dir}"

for d in "${target_dir}" "${draft_dir}"; do
  [ -f "${d}/config.json" ] || die "${d} has no config.json — download incomplete?"
  ls "${d}"/*.safetensors >/dev/null 2>&1 || die "${d} has no .safetensors files"
done

modelfile="${WORKDIR}/Modelfile"
{
  echo "FROM ${target_dir}"
  echo "DRAFT ${draft_dir}"
} > "${modelfile}"
echo ">> wrote ${modelfile}:"
sed 's/^/   /' "${modelfile}"

create_args=("${MODEL_NAME}" -f "${modelfile}" --quantize "${QUANTIZE}")
[ -n "${DRAFT_QUANTIZE}" ] && create_args+=(--draft-quantize "${DRAFT_QUANTIZE}")

echo ">> ollama create ${create_args[*]}"
ollama create "${create_args[@]}"

cat <<EOF

Done. The drafter is a separate gemma4_assistant GGUF, attached at serve time.
MTP helps at temperature=0 (greedy). Verify with:

  curl http://localhost:11434/api/generate -d '{"model":"${MODEL_NAME}","prompt":"What is 2+2?","stream":false,"options":{"temperature":0,"num_ctx":4096}}'

In the llama-server logs, look for: --spec-type gemma4-mtp --mtp-head <path>
and the assistant loading as gemma4_assistant.
EOF
