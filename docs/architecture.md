# Architecture

This document grows with the implementation. The backend section below is the result of the Phase 1 spike.

## Backend

### Decision (proposed and approved 2026-11-15)

Each worker runs its layer range with **our own Llama-family stage engine written against ggml's public API** (`ggml.h`, `ggml-backend.h`, `gguf.h`). One GGUF file serves every backend: ggml-CPU, ggml-Metal and ggml-CUDA, with Vulkan possible later. The HF transformers fp32 CPU model is the correctness oracle. MLX and ONNX Runtime are not used.

### Evidence

All measurements use TinyLlama-1.1B-Chat (HF revision `fe8a4ea1`), converted to GGUF with llama.cpp `b11960` (commit `8642541b`). Greedy decoding of 32 tokens on one chat prompt (27 prompt tokens); logits are compared on every row.

The machines:
- Mac: Apple M5, 16 GB, macOS 27.2.1.
- herb: HP Omen 16, RTX 3050 Laptop 4 GB, driver 595.99.02, CUDA 12.8, Ubuntu 24.04.
- The link between them: Wi-Fi with Tailscale. The LAN path is blocked by herb's firewall.

**Correctness** (`spike/compare.py`, reference = HF fp32 CPU):

| Engine run | Greedy tokens | Top-1 | Max abs Δlogit | KL mean / max |
|---|---|---|---|---|
| CPU, F32 weights, one stage | 32/32 equal | 1.0000 | 1.3e-4 | 7.4e-12 / 1.2e-10 |
| Metal, F16 weights | 32/32 | 1.0000 | 4.3e-2 | 2.7e-6 / 1.2e-4 |
| Metal, F32 weights | 32/32 | 1.0000 | 4.5e-2 | 2.3e-6 / 9.6e-5 |
| CUDA, F16 weights | 32/32 | 1.0000 | 1.9e-1 | 1.8e-5 / 3.3e-4 |
| Metal [0,11) → CPU [11,22), F16 | 32/32 | 1.0000 | 2.0e-1 | 1.1e-5 / 1.9e-4 |
| CUDA [0,11) → CPU [11,22), F16 | 32/32 | 1.0000 | 8.8e-1 | 3.3e-4 / 1.0e-2 |
| Metal F16 vs CUDA F16 (no HF) | 32/32 | 1.0000 | 2.0e-1 | 2.3e-5 / 3.4e-4 |

On a fixed backend, two stages [0,11) + [11,22) are **bit-exact** with one stage [0,22). This holds on CPU, Metal and CUDA.

**Decode speed**, F16 weights, warmed up, in tokens/s:

| Device | Our engine | llama.cpp `llama-bench tg128` | MLX (`mlx-lm` 0.32.2) |
|---|---|---|---|
| M5, Metal | 57.3 | 58.4 | 58.4 to 59.1 |
| RTX 3050 Laptop, CUDA | 75.5 | 83.0 | n/a |

The gap on CUDA has known causes that llama.cpp already addresses: CUDA graphs, flash attention and an F16 KV cache. Prefill was not measured properly: the first forward call includes backend warm-up.

**Two machines.** llama.cpp RPC (`ggml-rpc-server` on herb, Metal plus RPC CUDA, layer split 1:1) over this link decodes **22.1 ± 6.3 t/s**, slower than either device alone. A TCP round trip carrying one 8 KB activation over the same link measures p50 9.1 ms, p90 12.3 ms, p99 19.9 ms (`spike/tcp_rtt.py`). One RPC token therefore costs several round trips.

A pipeline with one hop per stage boundary should cost about 8.7 ms (Metal, layers 0–10) + 6.6 ms (CUDA, layers 11–21 and head) + about 9 ms (one round trip), roughly 24 ms or ~40 t/s. **This is a projection, not a measurement.** The vertical slice measures it, and the pivot rule (ship a control plane over llama.cpp RPC if ours is ≥20% slower on the same setup) is evaluated then.

**Rejected alternatives:**
- **Patching llama.cpp to run layer ranges.** prima.cpp did this as a fork of llama.cpp's former single 23,760-line `src/llama.cpp`; its public mirror was last updated 2025-05-14. Upstream now has 70 source files and 170 architecture builders and renames tools between releases (`rpc-server` became `ggml-rpc-server`). Our engine is 326 lines on the public ggml API.
- **ONNX Runtime with the CoreML execution provider** (onnxruntime 1.31, Optimum export with KV cache). CoreML claimed 2186 of 2642 nodes in **133 partitions**, then failed to build the model (`error code: -7`), both with all compute units and with CPU and GPU only. The ORT CPU provider decodes at 18.7 t/s.
- **MLX.** No decode advantage at this size; decode is memory-bandwidth-bound on all three engines. It would add a second weight format.

### Proposed parity tolerances

These are derived from the table above and are to be confirmed on more prompts and a second model.
- **Same backend, stage split vs single process:** bit-exact.
- **CPU F32 vs HF fp32:** max |Δlogit| ≤ 1e-3.
- **Across backends and hardware:** top-1 agreement ≥ 0.99 and mean KL ≤ 1e-3 per row.

### Open questions from the spike

- The CUDA → CPU handoff diverges about 10× more than the other mixes (max KL 1e-2). The likely cause is activations being rounded to F16 inside the F16 matrix products on both sides. To be investigated before tolerances are fixed.
- Only F16/F32 weights were tested. Quantized GGUF (Q8_0, Q4_K) should run unchanged, since ggml handles it inside `mul_mat` and `get_rows`, but this has not been run.
- Llama 3 needs RoPE frequency factors (`rope_freqs.weight`) and Qwen2 needs QKV biases; neither is implemented.

### Reproducing

```bash
git clone --depth 1 --branch b11960 https://github.com/ggml-org/llama.cpp .cache/llama.cpp
uv sync --extra oracle
uv run --with ./.cache/llama.cpp/gguf-py --with sentencepiece python .cache/llama.cpp/convert_hf_to_gguf.py \
  ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6 \
  --outtype f16 --outfile .cache/models/tinyllama-1.1b-chat-f16.gguf
cmake -S spike/stage -B .cache/build-stage -G Ninja -DCMAKE_BUILD_TYPE=Release   # add -DGGML_CUDA=ON on NVIDIA
cmake --build .cache/build-stage
uv run --extra oracle python spike/reference.py --gen 32 --out .cache/runs/ref
.cache/build-stage/stage_probe --model .cache/models/tinyllama-1.1b-chat-f16.gguf --backends gpu,cpu --cuts 11 \
  --gen 32 --tokens "$(python3 -c 'import json;print(",".join(map(str,json.load(open(".cache/runs/ref.tokens"))["prompt"])))')" \
  --logits .cache/runs/test.bin
uv run python spike/compare.py .cache/runs/ref.bin .cache/runs/test.bin --prompt-len 27
```
