# Gemma 4 Multi-Token Prediction (MTP)

Gemma 4 models ship with companion MTP assistant models that accelerate inference by predicting multiple tokens per forward pass. Unlike traditional speculative decoding -- where a separate draft model generates tokens independently and the target model verifies them -- MTP assistants share the target model's KV cache and operate directly on its hidden states. This makes MTP significantly more memory-efficient and faster than running two independent models.

## Requirements

- NVIDIA GPU with CUDA support (e.g., RTX A6000, A100, etc.)
- Ollama built from this fork with `MLX_ENGINE=ON` (the internal MLX runtime, not to be confused with Apple's MLX Python framework -- Ollama's MLX engine has a CUDA backend for NVIDIA GPUs)
- The `--experimental` flag gates safetensors model creation
- Gemma 4 model weights in safetensors format (downloaded from HuggingFace)

## Quickstart

### 1. Download the model and assistant

Download both the target model and its MTP assistant from HuggingFace. The assistant model follows the naming convention `{model}-assistant`:

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

### 2. Create the model

If the assistant directory follows the `{model}-assistant` naming convention and sits in the same parent directory, Ollama auto-detects it:

```bash
ollama create --experimental mymodel -f Modelfile
```

Where `Modelfile` contains:

```
FROM ./gemma-4-27B-it
```

Ollama logs `detected MTP assistant model` when auto-detection succeeds.

### 3. Run it

```bash
ollama run mymodel
```

MTP engages automatically during generation. No additional flags are needed at runtime.

## Explicit DRAFT directive

If the assistant model is in a different location or doesn't follow the naming convention, use the `DRAFT` directive in your Modelfile:

```
FROM ./gemma-4-27B-it
DRAFT /path/to/gemma-4-27B-it-assistant
```

The path must point to a directory containing `config.json` and `*.safetensors` files with a known assistant architecture (`Gemma4AssistantForCausalLM` or `gemma4_assistant` in the config).

## Quantizing the assistant model

You can quantize the assistant model independently from the target model using `--draft-quantize`:

```bash
ollama create --experimental -f Modelfile mymodel --draft-quantize mxfp8
```

This reduces assistant model memory usage while keeping the target model at full precision. Supported quantization types match the standard Ollama quantization options (e.g., `mxfp8`, `int4`, `int8`).

## How MTP works (vs. speculative decoding)

Traditional speculative decoding uses two fully independent models: a small "draft" model generates candidate tokens, then the larger target model verifies them in a single batched forward pass. Both models maintain separate KV caches and the draft model runs its own prefill.

Gemma 4 MTP is architecturally different:

1. **Shared KV cache.** The assistant reuses the target model's key-value cache. It runs query-only attention against the target's existing KV pairs, so there is no separate prefill and no duplicated memory for cached context.
2. **Hidden-state input.** The assistant receives the target model's token embeddings and hidden states through projection layers (`PreProjection` / `PostProjection`), not raw token IDs. This gives it richer signal with minimal compute.
3. **Lightweight architecture.** Assistant layers are simplified: query-only attention (no separate K/V projections) plus standard MLP blocks, with layer scalars for stability. The result is a much smaller model that still predicts accurately.

The net effect is faster inference with lower memory overhead than a traditional draft model of equivalent quality.

## Tuning MTP behavior

MTP is automatic when an assistant model is present, but you can tune it via environment variables. The `OLLAMA_MLX_` prefix refers to Ollama's internal runtime engine name, not Apple's MLX Python framework -- these variables work on CUDA/NVIDIA GPUs:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MLX_MTP_INITIAL_DRAFT_TOKENS` | Model-dependent (4-14) | Number of tokens the assistant predicts per iteration |
| `OLLAMA_MLX_MTP_MAX_DRAFT_TOKENS` | 16 | Upper bound on draft tokens per iteration |
| `OLLAMA_MLX_MTP_DRAFT_SCHEDULE` | `constant` | `constant` (fixed count) or `heuristic` (adapts based on acceptance rate) |

The default initial draft count varies by model size:

| Model | Initial draft tokens |
|---|---|
| Gemma 4 27B (dense, 60 layers) | 14 |
| Gemma 4 MoE (30 layers) | 8 |
| Other Gemma 4 variants | 4 |

Higher values generate more candidates per step. If the acceptance rate is high, this yields larger speedups; if low, the wasted compute increases. The `heuristic` schedule adjusts automatically.

## Supported models

MTP assistants are detected by their HuggingFace architecture field in `config.json`:

- `Gemma4AssistantForCausalLM`
- `gemma4_assistant` (via `model_type`)

The Gemma 4 target model itself must use `Gemma4ForCausalLM` or `Gemma4ForConditionalGeneration`.

## Common mistakes

**Using Apple's MLX Python framework.** The MTP assistant integration is built into this Ollama fork's runtime, which has its own CUDA backend for NVIDIA GPUs. Despite the internal naming (`mlxrunner`, `OLLAMA_MLX_*` env vars), this is **not** Apple's MLX Python framework and does not require macOS or Apple Silicon. Running the assistant model through standalone MLX Python scripts or other frameworks bypasses the shared KV cache mechanism and won't give you MTP -- you'll just be running two independent models. Use this Ollama fork on a CUDA-capable GPU.

**Forgetting `--experimental`.** The `DRAFT` directive and safetensors model creation require the `--experimental` flag on `ollama create`. Without it, you'll get an error.

**Treating MTP as speculative decoding.** MTP is not a drop-in for generic speculative decoding frameworks. The assistant model is architecturally coupled to Gemma 4's specific hidden-state format and KV cache layout. It won't work as a standalone draft model in other frameworks.

## Troubleshooting

### `quantization requires MLX support`

You're running `ollama create --experimental --quantize ...` but the MLX CUDA library wasn't built. The MLX engine is required for `--quantize` in the `--experimental` (safetensors) create path. Build with `cmake --preset 'MLX CUDA 13'` and install the MLX component. See [Building from source](#building-from-source-cuda) below.

**Workaround:** If you have pre-quantized weights (e.g., NVFP4 from HuggingFace), omit `--quantize` -- Ollama will import them as-is without needing the MLX quantizer.

### `MLX not available: failed to load MLX dynamic library`

The model was created successfully, but the runtime can't find the MLX CUDA shared library. Ensure:

1. The `MLX CUDA 13` preset was built and installed (see above)
2. The `mlx_cuda_v13/` directory exists under your Ollama lib path (e.g., `/usr/lib/ollama/mlx_cuda_v13/`)
3. CUDA drivers are properly installed and visible

### Model created but MTP not engaging

MTP only activates when:

- The model was created with an assistant model (check `ollama show mymodel` for draft metadata)
- Greedy decoding: `temperature=0`, no logprobs, no repeat/presence/frequency penalties
- Sampled decoding: `temperature>0` (with penalties allowed)

If using the API, set `"options": {"temperature": 0}` for greedy MTP or use any non-zero temperature for sampled MTP.

## Building from source (CUDA)

The standard Dockerfile builds CUDA and CPU runners but **does not** build the MLX engine by default. MTP requires the MLX engine (which has a CUDA backend -- the naming is historical). You must build the `MLX CUDA 13` preset in addition to the standard `CUDA 13` preset.

### Using CMake presets directly

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

Both components must be installed to `lib/ollama/`. The MLX build produces `mlx_cuda_v13/` inside that directory, which the mlxrunner loads at startup.

### Dockerfile

The stock `Dockerfile` in this repo does **not** include the MLX CUDA build. To get MTP working in Docker, add the MLX CUDA build step. After the existing CUDA build:

```dockerfile
RUN cmake --preset 'MLX CUDA 13' -DCMAKE_CUDA_ARCHITECTURES=86 \
    && cmake --build --preset 'MLX CUDA 13' -j$(nproc) \
    && cmake --install build --component MLX --strip
```

Without this step, you'll get `quantization requires MLX support` at create time and `MLX not available: failed to load MLX dynamic library` at inference time. These errors do not mean you need macOS or Apple Silicon -- they mean the MLX CUDA library was not built.

### Forcing a variant

You can force a specific MLX variant with the `OLLAMA_LLM_LIBRARY` environment variable:

```bash
export OLLAMA_LLM_LIBRARY=mlx_cuda_v13
```
