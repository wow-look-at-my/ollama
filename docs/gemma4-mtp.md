# Building the Gemma 4 MTP model

This fork serves Gemma 4 with **Multi-Token Prediction (MTP)** speculative
decoding. MTP needs two pieces:

1. the full Gemma 4 **target** model, and
2. a small **drafter** ("assistant") that proposes several tokens the target
   verifies in one pass.

In this fork the drafter is a **separate** GGUF with
`general.architecture = "gemma4_assistant"`. At serve time `llama-server` loads it
into the target via `--mtp-head` and it cross-attends the target's KV cache (so
its attention is query-only — no second KV cache). You produce it with a single
`ollama create` from the Hugging Face safetensors, using a Modelfile with a
`DRAFT` directive.

> The drafter is **not** embedded into the target GGUF. Bundling `draft.*` tensors
> into one file is the old, removed design and makes the target loader reject the
> file (`done_getting_tensors: wrong number of tensors`). Keep the drafter
> separate — the converter and `ollama create` below do this for you.

## TL;DR

```sh
# read-only HF token with access to the (gated) google/gemma-4 repos
export HF_TOKEN=hf_xxx
# this fork's ollama must be built and `ollama serve` running on localhost
scripts/build-gemma4-mtp.sh
```

That downloads the target + drafter, writes a Modelfile, and runs `ollama create`.
Defaults: model `gemma4-mtp`, target `google/gemma-4-31B-it`, drafter
`google/gemma-4-31B-it-assistant`, quantize `q4_K_M`. Override with env vars
(`MODEL_NAME`, `TARGET_REPO`, `DRAFT_REPO`, `QUANTIZE`, `DRAFT_QUANTIZE`,
`WORKDIR`).

## Can I just pull `gemma4:31b-coding-mtp-bf16` and quantize it?

No. That published artifact is **macOS/MLX-gated** (`HTTP 412: this model requires
macOS`) and is an **MLX** model — its MTP is baked into the MLX format, not the
separate `gemma4_assistant` GGUF this fork's `llama-server` loads via `--mtp-head`.
You must build from the Hugging Face safetensors as below.

## Prerequisites

- This fork's `ollama` binary built and on `PATH`
  (`go build -o /usr/bin/ollama .`), with `ollama serve` running on localhost.
- `HF_TOKEN` with read access to `google/gemma-4-31B-it` and
  `google/gemma-4-31B-it-assistant` (gated — accept the license on HF once).
- `hf` or `huggingface-cli` on `PATH` (`pip install -U huggingface_hub`).
- Disk: the target is ~60 GB at bf16; with the GGUF output and scratch space,
  leave ~100 GB free. The drafter is ~1 GB.

## Manual steps

If you'd rather not use the script:

1. Download both repos:
   ```sh
   hf download google/gemma-4-31B-it           --local-dir ~/gemma4-mtp/target
   hf download google/gemma-4-31B-it-assistant --local-dir ~/gemma4-mtp/draft
   ```
2. Write a `Modelfile`:
   ```
   FROM ~/gemma4-mtp/target
   DRAFT ~/gemma4-mtp/draft
   ```
   `FROM` is the target safetensors directory; `DRAFT` is the assistant
   safetensors directory. `ollama create` converts the target to GGUF and the
   drafter to a standalone `gemma4_assistant` GGUF (a `MediaTypeImageDraft`
   layer).
3. Create the model:
   ```sh
   ollama create gemma4-mtp -f Modelfile --quantize q4_K_M
   ```
   Add `--draft-quantize <level>` to quantize the drafter too (it is small, so
   this is optional).

   For supported quant types (`Q4_K_M` and `Q8_0`) the target is quantized
   **during conversion in a single streaming pass** — the safetensors are
   mmapped, quantized, and written straight to the final GGUF, with no
   full-precision intermediate file and no separate `llama-quantize` process. The
   output is byte-for-byte identical to running `llama-quantize` (the block
   kernels and the per-tensor k-quant mixture are validated against it). Other
   types (e.g. `Q4_K_S`, which the mixture would route partly through Q5_K) and
   multimodal models fall back to the old convert-then-`llama-quantize` path
   automatically.

## Verify

MTP only helps at **temperature 0** (greedy), where it is mathematically
equivalent to plain autoregressive decoding — identical output, just faster:

```sh
curl http://localhost:11434/api/generate -d '{
  "model":"gemma4-mtp","prompt":"What is 2+2?","stream":false,
  "options":{"temperature":0,"num_ctx":4096}
}'
```

In the `llama-server` logs, confirm the drafter is wired up: look for
`--spec-type gemma4-mtp --mtp-head <path>` and the assistant loading as
`gemma4_assistant`.

## Notes

- Tested on an RTX A6000 (49 GB). At `q4_K_M` + `num_ctx=4096` the target plus
  drafter use ~23.7 GiB of VRAM.
- The drafter carries the target's tokenizer (the runtime compares vocab text),
  and `gemma4_assistant.n_embd_backbone` must equal the target's embedding length
  (5376 for `gemma-4-31B-it`). The converter handles both automatically.
- Other sizes work too — point `TARGET_REPO`/`DRAFT_REPO` at the matching pair,
  e.g. `google/gemma-4-12B-it` + `google/gemma-4-12B-it-assistant`.
- Converter internals and the on-disk contract: see `convert/convert_gemma4.go`
  (`ConvertGemma4MTPDraft`) and the companion `wow-look-at-my/llama.cpp` fork's
  `CLAUDE.md`.
