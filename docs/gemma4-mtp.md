# Gemma 4 Multi-Token Prediction (MTP)

Gemma 4 models ship with companion MTP assistant models that accelerate inference by predicting multiple tokens per forward pass. Unlike traditional speculative decoding -- where a separate draft model generates tokens independently and the target model verifies them -- MTP assistants share the target model's KV cache and operate directly on its hidden states. This makes MTP significantly more memory-efficient and faster than running two independent models.

## Why `--experimental` (not a GGUF)

Standard Ollama converts models to GGUF and runs them through llama.cpp. MTP does not use that pipeline. The MTP inference runtime lives in `mlxrunner`, a separate backend built on the [MLX C library](https://github.com/ml-explore/mlx) (which, despite the name, has a CUDA backend for NVIDIA GPUs -- it is not macOS-only).

The `--experimental` flag on `ollama create` selects this alternative pipeline:

1. **Import:** Safetensors are imported directly (no GGUF conversion)
2. **Runtime:** Models route to `mlxrunner` instead of llama.cpp
3. **MTP:** The mlxrunner loads the assistant tensors and runs shared-KV-cache MTP

This is why you download safetensors from HuggingFace and use `--experimental`, rather than loading a pre-quantized GGUF. The GGUF/llama.cpp pipeline has no MTP implementation.

## Requirements

- NVIDIA GPU with CUDA support (e.g., RTX A6000, A100)
- Ollama built from this fork **with the `MLX CUDA 13` CMake preset** (see [Building from source](#building-from-source-cuda))
- Gemma 4 model weights in safetensors format from HuggingFace

## Quickstart

### 1. Download the model and assistant

Download both the target model and its MTP assistant from HuggingFace. The assistant follows the `{model}-assistant` naming convention:

```
models/
  gemma-4-27B-it/
    config.json
    model-00001-of-00006.safetensors
    ...
  gemma-4-27B-it-assistant/
    config.json
    model.safetensors
```

If you want to save VRAM, use pre-quantized weights (e.g., NVFP4 variants from HuggingFace). This avoids needing Ollama to quantize at create time.

### 2. Create the model

If the assistant directory follows the naming convention and sits next to the target, Ollama auto-detects it:

```bash
ollama create --experimental mymodel -f Modelfile
```

Where `Modelfile` is:

```
FROM ./gemma-4-27B-it
```

Ollama logs `detected MTP assistant model` when auto-detection succeeds. No `--quantize` flag is needed if your source weights are already quantized.

### 3. Run it

```bash
ollama run mymodel
```

MTP engages automatically during generation. No additional flags are needed at runtime.

## Explicit DRAFT directive

If the assistant model is in a different location or doesn't follow the naming convention, use the `DRAFT` directive:

```
FROM ./gemma-4-27B-it
DRAFT /path/to/gemma-4-27B-it-assistant
```

The path must point to a directory containing `config.json` and `*.safetensors` files with a known assistant architecture (`Gemma4AssistantForCausalLM` or `gemma4_assistant`).

## Optional: quantizing at create time

If your source weights are full precision (bf16) and you want to reduce VRAM usage, you can quantize during import. This requires the MLX CUDA library to be built.

Quantize the target model:

```bash
ollama create --experimental -f Modelfile mymodel --quantize mxfp8
```

Quantize only the assistant model (keeping the target at full precision):

```bash
ollama create --experimental -f Modelfile mymodel --draft-quantize mxfp8
```

Supported types in the experimental path: `mxfp8`, `int4`, `int8`, `nvfp4`. These are MLX quantization types, not GGML types (`q4_K_M`, `q8_0`, etc. will not work here).

**Recommended approach:** Skip this entirely and download pre-quantized weights from HuggingFace instead.

## How MTP works (vs. speculative decoding)

Traditional speculative decoding uses two fully independent models: a small "draft" model generates candidate tokens, then the larger target model verifies them in a single batched forward pass. Both models maintain separate KV caches and the draft model runs its own prefill.

Gemma 4 MTP is architecturally different:

1. **Shared KV cache.** The assistant reuses the target model's key-value cache. It runs query-only attention against the target's existing KV pairs, so there is no separate prefill and no duplicated memory for cached context.
2. **Hidden-state input.** The assistant receives the target model's token embeddings and hidden states through projection layers, not raw token IDs. This gives it richer signal with minimal compute.
3. **Lightweight architecture.** Assistant layers are simplified: query-only attention (no separate K/V projections) plus standard MLP blocks, with layer scalars for stability. The result is a much smaller model that still predicts accurately.

The net effect is faster inference with lower memory overhead than a traditional draft model of equivalent quality.

## Tuning MTP behavior

MTP is automatic when an assistant model is present, but you can tune it via environment variables. The `OLLAMA_MLX_` prefix refers to Ollama's internal runtime engine name -- these variables work on CUDA/NVIDIA GPUs:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MLX_MTP_INITIAL_DRAFT_TOKENS` | Model-dependent (4-14) | Tokens the assistant predicts per iteration |
| `OLLAMA_MLX_MTP_MAX_DRAFT_TOKENS` | 16 | Upper bound on tokens per iteration |
| `OLLAMA_MLX_MTP_DRAFT_SCHEDULE` | `constant` | `constant` or `heuristic` (adapts based on acceptance rate) |

Default initial draft tokens by model size:

| Model | Initial draft tokens |
|---|---|
| Gemma 4 27B (dense, 60 layers) | 14 |
| Gemma 4 MoE (30 layers) | 8 |
| Other Gemma 4 variants | 4 |

## Supported models

MTP assistants are detected by their HuggingFace architecture field in `config.json`:

- `Gemma4AssistantForCausalLM`
- `gemma4_assistant` (via `model_type`)

The target model must use `Gemma4ForCausalLM` or `Gemma4ForConditionalGeneration`.

## Common mistakes

**Using Apple's MLX Python framework.** Despite the internal naming (`mlxrunner`, `OLLAMA_MLX_*` env vars), this fork does **not** use Apple's MLX Python framework and does not require macOS. The MLX C library used here has a CUDA backend. Running the assistant model through standalone MLX Python scripts bypasses the shared KV cache mechanism and won't give you MTP. Use this Ollama fork.

**Trying to use GGUFs.** MTP is not available through the standard GGUF/llama.cpp pipeline. You must use `--experimental` with safetensors input. See [Why `--experimental`](#why---experimental-not-a-gguf) above.

**Using GGML quantization types.** The `--experimental` path uses MLX quantization (`mxfp8`, `int4`, `int8`), not GGML quantization (`q4_K_M`, `q8_0`). If you get `unsupported quantization type`, you're using GGML names. Better yet, skip quantization and use pre-quantized weights from HuggingFace.

**Treating MTP as speculative decoding.** MTP is not a drop-in for generic speculative decoding frameworks. The assistant model is architecturally coupled to Gemma 4's hidden-state format and KV cache layout.

## Troubleshooting

### `quantization requires MLX support`

The MLX CUDA library wasn't built. Either:
- Build with `cmake --preset 'MLX CUDA 13'` (see below), or
- Use pre-quantized weights and omit `--quantize`

### `MLX not available: failed to load MLX dynamic library`

The model was created but the MLX CUDA shared library isn't installed. Ensure:

1. The `MLX CUDA 13` preset was built and installed
2. `mlx_cuda_v13/` exists under your Ollama lib path (e.g., `/usr/lib/ollama/mlx_cuda_v13/`)
3. CUDA drivers are properly installed

### Model created but MTP not engaging

MTP activates when:

- The model has an assistant bundled (check `ollama show mymodel` for draft metadata)
- Greedy decoding: `temperature=0`, no logprobs, no repeat/presence/frequency penalties
- Sampled decoding: `temperature>0` (penalties allowed)

## Building from source (CUDA)

The standard Dockerfile builds CUDA and CPU runners but **does not** build the MLX engine. MTP requires the MLX engine. You must build the `MLX CUDA 13` preset in addition to `CUDA 13`.

### CMake presets

```bash
# Standard CUDA runner (for GGUF models)
cmake --preset 'CUDA 13'
cmake --build --preset 'CUDA 13'
cmake --install build --component CUDA --strip

# MLX CUDA runner (required for MTP)
cmake --preset 'MLX CUDA 13'
cmake --build --preset 'MLX CUDA 13'
cmake --install build --component MLX --strip
```

Both components install to `lib/ollama/`. The MLX build produces `mlx_cuda_v13/` inside that directory.

### Dockerfile

The stock `Dockerfile` does **not** include the MLX CUDA build. Add this after the existing CUDA build step:

```dockerfile
RUN cmake --preset 'MLX CUDA 13' -DCMAKE_CUDA_ARCHITECTURES=86 \
    && cmake --build --preset 'MLX CUDA 13' -j$(nproc) \
    && cmake --install build --component MLX --strip
```

Without this, you'll get `quantization requires MLX support` at create time and `MLX not available: failed to load MLX dynamic library` at runtime. These errors mean the MLX CUDA library wasn't built -- not that you need macOS.

### Forcing a variant

```bash
export OLLAMA_LLM_LIBRARY=mlx_cuda_v13
```
