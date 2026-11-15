# shardline

**Pipeline-parallel LLM serving across a LAN of mixed NVIDIA and Apple Silicon machines.**

shardline (formerly Hyperlane) aims to serve models that don't fit in any single machine's GPU memory. It splits them layer-wise across 2–4 machines, any mix of NVIDIA GPUs on Linux and Apple Silicon Macs, behind one OpenAI-compatible endpoint.

Pipeline parallelism adds capacity, not single-stream speed: one request keeps one stage busy at a time, so throughput gains need several requests in flight.

## Status

**Pre-alpha. A model runs split across workers from the command line; there is no serving API and no planner yet.** What exists today:

| Area | State |
|---|---|
| Worker (`worker/`, C++20) | Builds on macOS (arm64) and Linux. Runs a range of layers of a Llama-family GGUF with ggml, on the CPU and, with the `metal` preset, on the Apple GPU, and serves `shardline.v1.WorkerService` over gRPC: `GetStats` (host RAM, the Metal working-set limit capped by the RAM that is available since Metal and the CPU share it, and with `SHARDLINE_CUDA=ON` CUDA free/total memory), `LoadShard`, `Forward` and `CloseSession`. A model is named by its SHA-256 and read from `--model-dir`; a request cannot name a path. Binds `127.0.0.1` by default; there is no authentication yet |
| Model execution | One model splits across 1 to 3 worker processes on loopback and matches Hugging Face's fp32 model on TinyLlama: greedy tokens equal, fp32 on the CPU to 1.3e-4 in the logits and fp16 on Metal to 4.3e-2. Every split is bit-exact with a single worker on the same backend. Not yet measured across two machines |
| Discovery | Workers advertise `_shardline._tcp` over mDNS (`--advertise`, non-loopback listeners only) using the dns_sd API: native on macOS, Avahi's compat library on Linux. A worker that cannot register exits with an error, and one that loses the mDNS daemon registers again when it returns |
| Python package (`python/shardline`) | gRPC client, a `Pipeline` that relays hidden states between workers for greedy generation, and the `shardline stats`, `discover`, `model add` and `generate` commands; `discover` prints the first announced address that answers from the machine it runs on. Pure Python; torch is only in the `oracle` extra |
| Layer planner | The current partitioner assigns layers **non-contiguously**, which is invalid for a pipeline (recorded as a strict xfail test). A contiguous, throughput-aware planner replaces it |
| OpenAI API, queueing, micro-batching | Not implemented. Activations travel over gRPC streams, relayed hop by hop by the coordinator |
| CUDA | The device probe is built by the `worker-cuda` CI job (`nvidia/cuda` container). It has run on one NVIDIA GPU, an RTX 3050 Laptop, and reported 3.6 of 3.7 GiB free, matching `nvidia-smi` |

## Build

### macOS (Apple Silicon)

```bash
brew install cmake ninja grpc protobuf
cd worker
cmake --preset cpu && cmake --build --preset cpu && ctest --preset cpu
```

### Linux (Ubuntu 24.04)

```bash
sudo apt-get install -y cmake ninja-build g++ pkg-config \
  libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
  libavahi-compat-libdnssd-dev avahi-daemon
cd worker
cmake --preset cpu && cmake --build --preset cpu && ctest --preset cpu
```

Use `--preset cuda` instead of `cpu` to also report NVIDIA GPUs (requires the CUDA toolkit). On macOS, `--preset metal` also lets ggml run layers on the Apple GPU.

The first configure downloads ggml, a 38 MB archive pinned by SHA-256. Pass `-DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=<a llama.cpp checkout at b11960>` to use a local copy, or `-DSHARDLINE_GGML=OFF` to build a worker that reports devices but cannot run layers.

### Python

```bash
uv sync
uv run pytest                     # unit tests
SHARDLINE_WORKER_BIN=worker/build/cpu/shardline-worker uv run pytest   # plus tests against the real worker
```

The protobuf stubs are generated when the package is installed, so after the proto changes run `uv sync --reinstall-package shardline`. A test fails with this hint when they are stale.

## Try it

```bash
worker/build/cpu/shardline-worker --listen 127.0.0.1:50051 &
uv run shardline stats 127.0.0.1:50051
```

To run a model, put a Llama-family GGUF in the model directory and generate through a worker. The worker opens `<sha256>.gguf` from its `--model-dir` when asked to load, so a model can be added after the worker starts:

```bash
DIGEST=$(uv run shardline model add model.gguf)         # copies it into ~/.cache/shardline/models
uv run shardline generate --model $DIGEST --stage 127.0.0.1:50051 \
  --tokens 1,15043,3186 --max-new-tokens 16
```

Give `--stage` once per worker and `--cuts` the layers where the later ones start to split the model, for example `--stage A:50051 --stage B:50051 --cuts 11`. Prompts are token ids; there is no tokenizer yet.

`scripts/e2e_smoke.sh --backend cpu` checks the whole path end to end. It builds a random 4-layer Llama, starts workers, splits it over one, two and three of them, and compares the answers with a single worker (bit for bit) and with Hugging Face. `--model tinyllama` does the same with the spike's TinyLlama files in `.cache`, and `--backend metal` runs it on the Apple GPU with a worker built from the `metal` preset.

To make a worker discoverable from other machines, run it with `--listen 0.0.0.0:50051 --advertise` and use `uv run shardline discover`. This exposes an unauthenticated control port to your network, so only do it on a network you trust. On macOS the application firewall asks whether `shardline-worker` may accept incoming connections.

## Repository layout

```
proto/shardline/v1/   gRPC protocol (checked by buf lint and buf breaking)
worker/               C++ worker: CMake presets, sources, GoogleTest tests
python/shardline/     Python client, pipeline, discovery and CLI; stubs generated at build time
scripts/              parity oracles, the random-model builder and the end-to-end test
spike/                the Phase 1 backend measurements, kept as evidence
tests/                pytest suite
```

## Roadmap

1. Backend spike and vertical slice: done (TinyLlama generating across worker processes, matching a single process exactly on the same backend), apart from the transport benchmark against raw TCP and a measurement across two machines
2. Contiguous planner that balances by measured throughput and usable (not total) memory
3. `shardline serve` with an OpenAI-compatible API, queueing and micro-batching
4. Health checks, metrics, failure detection, graceful drain
5. mTLS cluster join, content-addressed weight delivery, Docker image for Linux/NVIDIA, Homebrew tap for macOS
6. Reproducible benchmarks against a single device and existing distributed runtimes

## License

[Apache-2.0](LICENSE)
