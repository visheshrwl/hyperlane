# shardline

**Pipeline-parallel LLM serving across a LAN of mixed NVIDIA and Apple Silicon machines.**

shardline (formerly Hyperlane) aims to serve models that don't fit in any single machine's GPU memory. It splits them layer-wise across 2–4 machines, any mix of NVIDIA GPUs on Linux and Apple Silicon Macs, behind one OpenAI-compatible endpoint.

Pipeline parallelism adds capacity, not single-stream speed: one request keeps one stage busy at a time, so throughput gains need several requests in flight.

## Status

**Pre-alpha. It cannot run a model yet.** What exists today:

| Area | State |
|---|---|
| Worker (`worker/`, C++20) | Builds CPU-only on macOS (arm64) and Linux. Serves `shardline.v1.WorkerService/GetStats` over gRPC, reporting host RAM, the Metal working-set limit on Apple Silicon (capped by the RAM that is available, since Metal and the CPU share it) and, when built with `SHARDLINE_CUDA=ON`, CUDA free/total memory. Binds `127.0.0.1` by default; there is no authentication yet |
| Discovery | Workers advertise `_shardline._tcp` over mDNS (`--advertise`, non-loopback listeners only) using the dns_sd API: native on macOS, Avahi's compat library on Linux. A worker that cannot register exits with an error, and one that loses the mDNS daemon registers again when it returns |
| Python package (`python/shardline`) | gRPC client and `shardline stats` / `shardline discover` CLI; `discover` prints the first announced address that answers from the machine it runs on. Pure Python; torch is only in the `oracle` extra |
| Layer planner | The current partitioner assigns layers **non-contiguously**, which is invalid for a pipeline (recorded as a strict xfail test). A contiguous, throughput-aware planner replaces it |
| Model execution, activation transport, OpenAI API | Not implemented. The planned engine is ggml, with gRPC streaming between stages |
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

Use `--preset cuda` instead of `cpu` to also report NVIDIA GPUs (requires the CUDA toolkit).

### Python

```bash
uv sync
uv run pytest                     # unit tests
SHARDLINE_WORKER_BIN=worker/build/cpu/shardline-worker uv run pytest   # plus tests against the real worker
```

## Try it

```bash
worker/build/cpu/shardline-worker --listen 127.0.0.1:50051 &
uv run shardline stats 127.0.0.1:50051
```

To make a worker discoverable from other machines, run it with `--listen 0.0.0.0:50051 --advertise` and use `uv run shardline discover`. This exposes an unauthenticated control port to your network, so only do it on a network you trust. On macOS the application firewall asks whether `shardline-worker` may accept incoming connections.

## Repository layout

```
proto/shardline/v1/   gRPC protocol (checked by buf lint and buf breaking)
worker/               C++ worker: CMake presets, sources, GoogleTest tests
python/shardline/     Python client, discovery and CLI; stubs generated at build time
tests/                pytest suite
```

## Roadmap

1. Backend spike and vertical slice: TinyLlama generating across two worker processes, matching a single-process reference exactly on the same backend
2. Contiguous planner that balances by measured throughput and usable (not total) memory
3. `shardline serve` with an OpenAI-compatible API, queueing and micro-batching
4. Health checks, metrics, failure detection, graceful drain
5. mTLS cluster join, content-addressed weight delivery, Docker image for Linux/NVIDIA, Homebrew tap for macOS
6. Reproducible benchmarks against a single device and existing distributed runtimes

## License

[Apache-2.0](LICENSE)
