# Ollama Fork — Gemma 4 MTP

This is a fork of ollama (upstream `ollama/ollama`, merged up to `4d1b53e6`,
2026-07-21) that builds and serves **Gemma 4 with Multi-Token Prediction
(MTP)** speculative decoding, plus a set of create-path performance features.
See `AGENTS.md` for upstream's shared agent instructions.

## How Gemma 4 MTP works in this fork

GGUF models are served by the **C++ `llama-server`** subprocess
(`server/sched.go` routes all non-MLX models to `llm.NewLlamaServer`).
`llama-server` comes from the **`wow-look-at-my/llama.cpp` fork pinned by the
`LLAMA_CPP_VERSION` file** (a 40-hex commit SHA; see "llama.cpp pin" below).
Since the 2026-07 uprev, that fork tracks upstream llama.cpp master — which
has its own Gemma 4 MTP implementation — so this fork no longer carries a
private MTP dialect anywhere:

- the **drafter is a separate GGUF** with
  `general.architecture = "gemma4-assistant"` (upstream llama.cpp's
  `LLM_ARCH_GEMMA4_ASSISTANT`), stored as a `MediaTypeImageDraft` layer; it is
  **not** embedded in the target file;
- at serve time it is a standard llama.cpp MTP draft model:
  `llm/llama_server.go` `appendMTPDraftArgs` (upstream's function, unmodified)
  emits `--spec-type draft-mtp --spec-draft-n-max N
  --spec-draft-backend-sampling --spec-draft-model <path>` whenever the model
  manifest carries a draft layer (`Model.DraftPath`);
- the assistant shares the target's KV cache (Q-only attention) and consumes
  the target's last-layer activations through `nextn.pre_projection` /
  `nextn.post_projection`;
- it must carry the **target's tokenizer** (the runtime compares vocab type
  and token text).

> **Old dialects are dead.** The fork-private `gemma4_assistant` (underscore)
> arch served via `--mtp-head`, and the even older embedded `draft.*` design,
> no longer load. On-disk models built with them must be re-created with
> `ollama create` (see `docs/gemma4-mtp.md`).

### create path (the part this fork owns)

`ollama create` with a gemma4 base + `DRAFT` directive produces the drafter.
`server/create.go:convertMTPDraftFromSafetensors` dispatches on the base
architecture: `gemma4` → `convert.ConvertGemma4MTPDraft` (this fork's
converter, the only gemma4-from-safetensors drafter path anywhere — upstream's
DRAFT support is qwen3.5-only); qwen → upstream's
`convert.ConvertQwen35MTPDraft`.

`ConvertGemma4MTPDraft` (`convert/convert_gemma4.go`) emits **upstream
llama.cpp's gemma4-assistant GGUF dialect**. The authoritative contract is the
pinned llama.cpp tree — `src/models/gemma4-assistant.cpp` (required hparams +
tensors) and `src/llama-arch.cpp` (names), with `conversion/gemma.py
Gemma4AssistantModel` as the reference converter. Key points the converter
implements (each is load-bearing; the loader hard-fails or silently
mis-RoPEs without them):

- KV prefix `gemma4-assistant.`; `embedding_length_out` = the target's
  `embedding_length` (loader **requires** it to differ from the assistant's
  own `embedding_length`); `nextn_predict_layers` = `block_count` (loader
  asserts equality).
- `attention.key_length`/`value_length` (global head dim) **and**
  `key_length_swa`/`value_length_swa` (sliding head dim) — the swa pair,
  `layer_norm_rms_epsilon`, `sliding_window`, `context_length`, and the
  `sliding_window_pattern` `[]bool` (len = block_count) are required reads.
- `rope.dimension_count` (= global head dim) **and** `rope.dimension_count_swa`
  (= sliding head dim) are always emitted — omitting the swa one silently
  defaults `n_rot_swa` to the global value.
- Per-layer `attention.head_count_kv` array when global vs sliding KV head
  counts differ.
- Tensors: `nextn.pre_projection.weight` `{2*n_embd_out, n_embd}`,
  `nextn.post_projection.weight` `{n_embd, n_embd_out}`, one global
  `rope_freqs.weight` `{global_head_dim/2}` (freq-factors 1.0 for rotated
  dims, 1e30 for the rest; emitted only when a full-attention layer exists),
  and `blk.%d.layer_output_scale.weight` **required on every layer** (from HF
  `layer_scalar`).
- The centroid LM head is **dropped entirely** (`masked_embedding.*` tensors
  skipped; no `use_ordered_embeddings`/`n_centroids`/... KVs): the upstream
  loader ignores only its own `masked_embd_*` names and hard-fails on any
  other unexpected tensor.
- `tokenizer.*` is copied from the converted target, but `tokenizer.ggml.model`
  is baked to `"gemma4"` (BPE): the Ollama target GGUF writes `"llama"` and
  relies on the `llama/compat` shim to flip it at load — that shim only runs
  for arch `gemma4`, and the assistant loads through the stock llama.cpp path.
- Pairing validation: the assistant's `hidden_size` must differ from the
  target width, and a declared `backbone_hidden_size` must match the paired
  target. `token_embd` OOV padding rows are trimmed to the tokenizer size.

Unit coverage: `convert/convert_gemma4_assistant_test.go` (round-trips a
synthetic checkpoint through the converter and asserts the dialect, including
negative coverage: no centroid emission, `embedding_length_out !=
embedding_length`, backbone-mismatch rejection).

### quantize path (fused single-pass)

`ollama create --quantize` quantizes the target **during conversion** for
supported types (`Q4_K_M`, `Q8_0`): `convertFromSafetensors` calls
`convert.ConvertModelQuantized`, which mmaps the safetensors, quantizes each
tensor, and writes the final GGUF directly — no full-precision intermediate
and no separate `llama-quantize` subprocess. `createModel` sees the layer is
already the requested type and skips the llama-quantize rewrite.

The Go kernels (`convert/quantize.go`) and the per-tensor k-quant **mixture**
(`convert/quantize_mixture.go`, a port of llama.cpp's `llama_tensor_get_type`
including `weight_name_comparer` and `use_more_bits`) are **byte-for-byte
identical** to `llama-quantize` — proven by `TestQuantizeMatchesGGML`
(kernels vs `quantize_row_*_ref`) and `TestFusedMixtureMatchesLlamaQuantize`
(mixture vs the real binary), both re-validated against the pinned llama.cpp
after the 2026-07 uprev. Anything not fusable (other types like `Q4_K_S` that
need Q5_K, projector splits, non-safetensor sources) returns
`convert.ErrFusedUnsupported` and falls back to convert-then-`llama-quantize`
(`server/quantization.go`). `nearestInt` (round-half-to-even) is mandatory —
`math.Round` silently diverges. The Q4_K/Q6_K scale searches have **AVX2**
inner loops (`convert/simd_amd64.s`, dispatched behind `cpu.X86.HasAVX2`,
scalar fallback in `simd_other.go`) that stay byte-identical.

To run the bit-identical tests, build the llama.cpp fork (at the pinned SHA)
and set `OLLAMA_GGML_LIB_DIR`/`OLLAMA_GGML_SRC_DIR`/`OLLAMA_GGML_INC_DIR`
(kernels) and `OLLAMA_LLAMA_QUANTIZE` (mixture) to it.

Related fork deviation: `fs/ggml/gguf.go` `WriteGGUF` keeps its write pool at
`GOMAXPROCS` (upstream caps plain byte-copy writes at 2) because in this fork
a tensor's `WriteTo` may carry the fused quantize compute.

### create input hashing (source mode)

For **local** create (`cmd.go` → `server.CreateDirect`, `sourceMode=true`) of
safetensors, input hashing is deferred and overlapped: `cmd.go` enumerates
files without hashing (`parser.CreateRequest(dir, false)`);
`convertFromSafetensors` hashes each input in a goroutine
(`server/blobhash.go:sha256File`) **concurrently with `WriteGGUF`'s quantize
pass** (same pages as the conversion's mmap — read from disk once), then
stages blobs via `EnsureBlobFromPath`. The remote/HTTP path and GGUF imports
keep the up-front content-addressed flow.

## llama.cpp pin

`LLAMA_CPP_VERSION` holds a **40-hex commit SHA** of `wow-look-at-my/llama.cpp`
(currently the upstream-uprev line). The build machinery
(`llama/server/CMakeLists.txt`, `cmake/local.cmake`) auto-disables shallow
clones for SHA pins; `scripts/update-llamacpp-version.sh` **keeps** a
checked-in SHA pin (a branch name still re-resolves; advance deliberately
with `LLAMA_CPP_REF=master scripts/update-llamacpp-version.sh`). CI
(docker-build.yaml) builds exactly the pinned SHA.

`llama/compat/` carries the load-time compatibility layer for existing
published Ollama GGUFs plus `001-llama-cpp-hooks.patch` (call-site hooks),
applied to the fetched llama.cpp at build time. **After every llama.cpp pin
advance, verify the patches still apply** (`test.yaml`'s `patches` job, or
locally `git apply --check` against the pinned tree) and regenerate per
`llama/compat/README.md` if needed. The hooks patch includes a fork-specific
hunk (the `meta_borrowed` guard) for the pinned loader's zero-copy metadata
path.

History note: at the earlier pin `d98acbd2` `llama-quantize` segfaulted on
any mmap-parsed input (`gguf_set_kv` read `kv.data_string[j]` directly, which
is empty for borrowed zero-copy contexts), breaking the **fallback** quantize
path. The one-line borrow-aware fix (`kv.get_val<std::string>(j)`) ships at
the current pin (`f2aaaa6a`+); the bit-identical mixture test doubles as the
regression check for it.

## GPU-only fork

CPU inference is disabled (`llm/llama_server.go`): with default `num_gpu`
(-1) the runner refuses to start without a GPU (`hasGPUDevice`) and fails the
load unless every layer offloaded (`verifyFullGPUOffload`,
`ErrCPUFallbackDisabled`); an explicit `num_gpu >= 0` bypasses. The
Dockerfile uses a vanilla `almalinux:8` base instead of the ROCm SDK image
(ROCm build dropped; stages remain untargeted).

## Key files

- `docs/gemma4-mtp.md` — getting-started guide (download → Modelfile → create
  → verify, including the GPU-box verification steps)
- `scripts/build-gemma4-mtp.sh` — one-command build (HF download + Modelfile +
  `ollama create`)
- `convert/convert_gemma4.go` — `ConvertGemma4MTPDraft` (+
  `gemma4AssistantModel`): standalone `gemma4-assistant` drafter GGUF;
  `gemma4Model`: the gemma4 target converter
- `convert/convert_gemma4_assistant_test.go` — converter unit tests
- `convert/quantize.go`, `convert/quantize_mixture.go`,
  `convert/simd_amd64.{s,go}` + `simd_other.go` — fused quantize kernels,
  k-quant mixture, AVX2 loops
- `convert/quantize_hash_test.go`, `convert/quantize_mixture_test.go` —
  bit-identical validation vs the pinned llama.cpp build
- `convert/quantize_test.go` + `convert/ggmlbench/` — speed benchmarks (cgo
  bench behind tag `ggmlbench`)
- `server/create.go` — `convertMTPDraftFromSafetensors` dispatch; fused
  quantize + deferred-hashing wiring; `server/blobhash.go`
- `server/quantization.go` — `llama-quantize` fallback path
- `llm/llama_server.go` — upstream `appendMTPDraftArgs`; GPU-only enforcement;
  load/prompt-progress parsing (see `llama/compat/README.md` for the current
  load-progress status)
- `llama/compat/` — published-GGUF compatibility layer + hooks patch
- `LLAMA_CPP_VERSION`, `scripts/update-llamacpp-version.sh` — llama.cpp pin
- llama.cpp fork: `src/models/gemma4-assistant.cpp`, `src/llama-arch.cpp`,
  `conversion/gemma.py` — the on-disk contract this fork's converter targets
- `x/create/create.go:DetectAssistantDir` — MLX-path auto-detection of a
  companion `*-assistant` checkpoint dir when no `DRAFT` directive is given

## Hardware

RTX A6000 (49GB VRAM). Model at Q4_K_M + num_ctx=4096 uses ~23.7 GiB.

## Historical note (superseded designs)

Earlier fork generations implemented MTP (1) in a pure-Go runner with
`draft.*` tensors embedded in the target GGUF, then (2) as the fork-private
`gemma4_assistant` llama.cpp arch served via `--mtp-head`. Both are gone —
upstream deleted the Go engine (`model/models/`, `kvcache/`), and the 2026-07
llama.cpp uprev replaced the private dialect with upstream's implementation.
Treat any notes referencing them as history.
