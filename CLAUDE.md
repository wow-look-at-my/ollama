# Ollama Fork — Gemma 4 MTP

## Previous conversation

Read `2026-05-28-045922-previous-conversation.txt` for full context on what's been done, what crashed, what was fixed, and what's left. It covers the entire history of wiring the DRAFT directive, fixing CUDA crashes, and testing MTP end-to-end.

This is a fork of ollama with Multi-Token Prediction (MTP) support for Gemma 4.

## Dev Container

`docker-compose.yml` defines a dev container (`ollama-dev`) with CUDA 13 devel, Go 1.26, cmake/ninja/ccache, and Claude Code. Source is bind-mounted at `/ollama`.

### First time setup inside the container

1. Build ggml CUDA backend (results cached in `ggml-build` volume):
   ```
   cmake --preset 'CUDA 13' -DCMAKE_CUDA_ARCHITECTURES=86
   cmake --build --preset 'CUDA 13' -j$(nproc)
   cmake --install build --component CUDA --strip
   cmake --preset CPU
   cmake --build --preset CPU -j$(nproc)
   cmake --install build --component CPU --strip
   ```

2. Build ollama: `go build -o /usr/bin/ollama .`

3. Create model (only needed once, stored in `ollama-data` volume):
   ```
   ollama serve &
   ollama create gemma4-mtp -f /models/Modelfile --quantize q4_K_M
   ```

### Iterating on Go code

`go build -o /usr/bin/ollama .` then restart `ollama serve`. No Docker rebuild needed.

### Testing MTP

```
curl http://localhost:11434/api/generate -d '{"model":"gemma4-mtp","prompt":"What is 2+2?","stream":false,"options":{"temperature":0,"num_ctx":4096}}'
```

MTP only helps at temperature=0 (greedy). In the `llama-server` logs, confirm the
drafter is wired up: look for the `--spec-type gemma4-mtp --mtp-head <path>` flags
and the assistant model loading (`gemma4_assistant`). (The old Go-runner
"MTP eligible"/"MTP accepted" log lines no longer exist — that engine was removed.)

### Re-create the model after the converter changes
The on-disk `gemma4-mtp` (built 2026-05-28) is the broken embedded build and cannot
be salvaged. After this fork's converter lands and deploys, re-create it so the
drafter becomes a separate `gemma4_assistant` GGUF:

```
ollama create gemma4-mtp -f /models/Modelfile --quantize q4_K_M
```

The `/models/Modelfile` is `FROM <gemma-4 target>` + `DRAFT <gemma-4 *-assistant>`
(both safetensors dirs). For a from-scratch build (download the HF safetensors,
write the Modelfile, run create) use the one-command helper:

```
HF_TOKEN=hf_xxx scripts/build-gemma4-mtp.sh
```

See `docs/gemma4-mtp.md` for the full getting-started guide (prerequisites,
manual steps, verification, and why the published macOS/MLX `gemma4:31b-coding-mtp-bf16`
cannot be reused here).

## How Gemma 4 MTP works in this fork (current architecture)

GGUF models are served by the **C++ `llama-server`** subprocess (`server/sched.go`
routes all non-MLX models to `llm.NewLlamaServer`). The pure-Go MTP runner that
earlier sessions wrote (`runner/ollamarunner/mtp.go`) was **removed in the uprev**;
`model/models/gemma4/*.go` remains in-tree but is **dead code for GGUF serving**.

MTP therefore runs entirely through the vendored llama.cpp fork (pinned by
`LLAMA_CPP_VERSION`, the `wow-look-at-my/llama.cpp` master). That fork adds a
`gemma4_assistant` architecture — the port of Google's Gemma 4 drafter:

- the **drafter is a separate GGUF** (`general.architecture = "gemma4_assistant"`),
  loaded via `--mtp-head <file>` and attached to the target
  (`llama_model_load_mtp_from_file`), not embedded in the target file;
- it is **Q-only** (cross-attends the target's KV cache), uses the target's
  last-layer activations via `mtp.pre_projection`/`mtp.post_projection`, and has a
  centroid-routed LM head (`mtp.centroids`/`mtp.token_ordering`);
- it must carry the **target's tokenizer** (the loader compares vocab text).

### create path (the part this fork owns)
`ollama create` with a gemma4 base + `DRAFT` directive produces that separate
drafter. `server/create.go:convertMTPDraftFromSafetensors` dispatches on the base
architecture: `gemma4` → `convert.ConvertGemma4MTPDraft` (emits a standalone
`gemma4_assistant` GGUF as a `MediaTypeImageDraft` layer); qwen → the unchanged
`convert.ConvertQwen35MTPDraft`. At serve time `llm/llama_server.go` detects the
`gemma4_assistant` draft layer and emits `--spec-type gemma4-mtp --mtp-head`.

> Embedding the draft into the target GGUF (the old `ConvertModelWithDraft`
> `draft.*` shape) is WRONG for this runtime: the target loader rejects the extra
> tensors (`done_getting_tensors: wrong number of tensors`). Keep the drafter
> separate.

### quantize path (fused single-pass)
`ollama create --quantize` quantizes the target **during conversion** for
supported types (`Q4_K_M`, `Q8_0`): `convertFromSafetensors` calls
`convert.ConvertModelQuantized`, which mmaps the safetensors, quantizes each
tensor, and writes the final GGUF directly — no full-precision intermediate and
no separate `llama-quantize` subprocess. `createModel` sees the layer is already
the requested type and skips the llama-quantize rewrite. This is the fast path;
it avoids materializing a ~2-bytes/param intermediate and re-reading it.

The Go kernels (`convert/quantize.go`) and the per-tensor k-quant **mixture**
(`convert/quantize_mixture.go`, a port of llama.cpp's `llama_tensor_get_type`,
including the `weight_name_comparer` ordering and `use_more_bits`) are
**byte-for-byte identical** to `llama-quantize` — proven by
`TestQuantizeMatchesGGML` (kernels vs `quantize_row_*_ref`) and
`TestFusedMixtureMatchesLlamaQuantize` (mixture vs the real binary). Anything not
fusable (other types like `Q4_K_S` that need Q5_K, multimodal projector splits,
non-safetensor tensor sources) returns `convert.ErrFusedUnsupported` and falls
back to the convert-then-`llama-quantize` path. `nearestInt` (ggml's
round-half-to-even) is mandatory for the K-quants — `math.Round` silently
diverges. To run the bit-identical tests, build the llama.cpp fork's ggml and
point `OLLAMA_GGML_LIB_DIR`/`_SRC_DIR`/`_INC_DIR` (kernels) and
`OLLAMA_LLAMA_QUANTIZE` (mixture) at it.

### Historical note (superseded)
Earlier sessions implemented MTP in the now-deleted Go runner (ForwardMTP /
MTPDraft / runMTPCycle, draft `draft.*` tensors embedded in one GGUF) and recorded
acceptance/throughput numbers against it. That engine no longer runs for GGUF;
treat those notes and `2026-05-28-045922-previous-conversation.txt` as history, not
the current design.

## Key Files

- `docs/gemma4-mtp.md` — getting-started guide (download → Modelfile → create → verify)
- `scripts/build-gemma4-mtp.sh` — one-command build (HF download + Modelfile + `ollama create`)
- `convert/convert_gemma4.go` — `ConvertGemma4MTPDraft` (+ `gemma4AssistantModel`): standalone `gemma4_assistant` drafter GGUF
- `convert/convert_gemma4_assistant_test.go` — converter unit tests (synthetic fixtures)
- `convert/quantize.go` — fused Go quant kernels (Q8_0/Q4_K/Q6_K, `nearestInt`)
- `convert/quantize_mixture.go` — `ConvertModelQuantized` driver + the k-quant type mixture + `ErrFusedUnsupported`
- `convert/quantize_hash_test.go`, `convert/quantize_mixture_test.go` — bit-identical kernel + mixture validation vs llama-quantize
- `convert/convert.go` — `convertModel` shared core (full-precision vs fused)
- `server/create.go` — `convertMTPDraftFromSafetensors` dispatch (gemma4 vs qwen); fused-quantize wiring in `convertFromSafetensors`
- `server/quantization.go` — `llama-quantize` fallback path + clean progress capture
- `llm/llama_server.go` — `gemma4_assistant` draft detection, `--mtp-head` / `--spec-type gemma4-mtp`
- llama.cpp fork: `src/models/gemma4-assistant.cpp`, `src/llama-arch.cpp` (arch + `mtp.*` tensors + `gemma4_assistant.*` KV)
- `model/models/gemma4/*.go`, `convert/convert.go:ConvertModelWithDraft` — the superseded Go-engine/embedded path (dead code; do not extend)

## Hardware

RTX A6000 (49GB VRAM). Model at Q4_K_M + num_ctx=4096 uses ~23.7 GiB.
