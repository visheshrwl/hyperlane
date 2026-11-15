# Hyperlane

**Distributed GPU inference for heterogeneous prosumer hardware.**

Hyperlane aims to run LLMs that don't fit on one consumer GPU by splitting them layer-wise (pipeline parallelism) across several machines on a local network, with a single command on the user's side. Target hardware is NVIDIA GPUs on Linux and Apple Silicon Macs (integrated, unified-memory GPU), mixed freely in one cluster.

## Status

**Early prototype. It cannot run a model end to end yet.** The repository is an architectural scaffold: the component boundaries and most of the plumbing are sketched, but the core path (load shard → run layers → ship activations → next stage → sample token) is not implemented. The table below was produced by code review; the C++ worker has not been compiled or run on GPU hardware.

| Area | State |
|---|---|
| Python mDNS discovery (`discovery.py`) | Implemented; untested against a real worker. Expects one TXT key per field, which the worker does not publish yet |
| Layer partitioner (`KnapsackPartitioner`) | Implemented, but assigns layers *non-contiguously* (e.g. A: 0,1,2,5 / B: 3,6 / C: 4,7), which is invalid for a pipeline |
| ONNX shard export | Draft. Hard-codes hidden size 4096 and feeds only hidden states (no attention mask, rotary embeddings or KV cache) |
| Shard deployment | Not implemented. `LoadShard` is never called, and the path it would send is a temp file on the client machine, not on the worker |
| Python gRPC client | Stubs only; every RPC method is a `TODO` and no generated protobuf code is checked in |
| `AutoDistributedModel.generate()` | Placeholder; returns a fixed string |
| Worker gRPC server | Skeleton. `GetStats` returns hard-coded values; `ExecutePipeline` and `SetNextWorker` are no-ops; the service object is destroyed while the server is still running |
| ONNX execution | Session loading only; `run()` is a stub |
| TCP tensor data plane | Skeleton. No wire framing, and the send queue is never fed |
| INT4 activation quantization | Draft kernels with a fixed scale and no clamping; not validated for accuracy |
| Avahi service publishing | Draft; never creates its Avahi poll object, so it cannot publish as written |
| Worker build | Does not build as written (duplicate class definitions across `worker.cc` and the per-class files; C++ `Worker` collides with the gRPC-generated `Worker`) |
| Apple Silicon / macOS | Not supported yet. The worker's CMake requires CUDA and Avahi, and the transfer path is built on CUDA pinned memory and streams; none of these has a macOS equivalent in the code |
| CI | Lint and worker-build failures are ignored (`|| true`, `continue-on-error`); the wheel workflow targets a retired runner image |

Architecture notes live in [DEVELOPMENT.md](DEVELOPMENT.md).

## Architecture

Hyperlane is split into two components.

### hyperlane_worker (C++/CUDA)
Per-GPU-machine inference server:
- **Control plane**: gRPC service for shard loading and orchestration
- **Execution engine**: ONNX Runtime with the CUDA provider
- **Data plane**: TCP sockets for inter-worker activation transfer
- **Service discovery**: Avahi/mDNS registration with GPU stats in TXT records
- **Transport optimisation (draft)**: FP16→INT4 activation quantization, async CUDA streams, pinned-memory staging

### hyperlane_client (Python)
Orchestration library:
- **Discovery**: zeroconf browser for workers on the LAN
- **Sharding**: partitions transformer layers across workers by free VRAM
- **ONNX export**: `torch.onnx.export` per shard
- **gRPC client**: control of workers (stubs)
- **pybind11 bridge**: TCP tensor socket for sending the first stage's input

## Pipeline Parallelism

Models are partitioned **layer-wise** so only one activation tensor crosses the network per stage boundary:

```
Input → GPU A (Layers 1-10) → GPU B (Layers 11-20) → GPU C (Layers 21-32) → Output
```

Intended per-stage flow:
1. Run the stage's layers on the incoming activations
2. (Optional) quantize the output in VRAM
3. Async device-to-host copy into pinned memory
4. Send over TCP to the next stage
5. Next stage receives, copies host-to-device, dequantizes, and continues

Pipeline parallelism lets you run models larger than any single GPU's memory. It does **not** reduce single-stream latency, and a single request keeps only one stage busy at a time; throughput gains require multiple requests in flight.

## Installation (from source)

### System requirements
- Ubuntu 20.04+ (the worker is Linux/NVIDIA only today; macOS / Apple Silicon workers are planned)
- NVIDIA GPU, CUDA Compute Capability 6.0+ (build is currently pinned to sm_86/sm_89)
- CUDA Toolkit 11.8+, cuDNN 8.0+
- gRPC + protobuf, ONNX Runtime with CUDA provider, Avahi

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake \
  protobuf-compiler libprotobuf-dev \
  libgrpc++-dev protobuf-compiler-grpc \
  libavahi-client-dev libavahi-common-dev \
  python3-dev python3-pip

# CUDA / cuDNN: https://developer.nvidia.com/cuda-toolkit
pip install onnxruntime-gpu

git clone <repo> hyperlane
cd hyperlane
bash build.sh
```

> The worker does not currently build as written (see [Status](#status)). The Python package is not published to PyPI.

## Usage (intended)

### Start workers
```bash
./hyperlane_worker/build/hyperlane_worker          # port 50051
./hyperlane_worker/build/hyperlane_worker 50052    # another port
```

### Discover workers and shard a model
```python
import time
from hyperlane_client import DiscoveryManager, AutoDistributedModel

discovery = DiscoveryManager()
discovery.start_discovery()
time.sleep(2)  # let mDNS responses arrive
print(f"Found {len(discovery.get_workers())} workers")

# Call from synchronous code: from_pretrained() runs its own event loop.
model = AutoDistributedModel.from_pretrained("meta-llama/Llama-2-7b-hf", discovery)

# Placeholder today: returns a fixed string, no inference happens.
print(model.generate("What is machine learning?", max_tokens=128))

discovery.stop_discovery()
```

## Repository layout

```
.
├── proto/service.proto          # gRPC service definition
├── hyperlane_worker/            # C++/CUDA worker (CMake)
│   ├── include/                 # worker, service_impl, onnx_session, tensor_sender/receiver, cuda_ops
│   └── src/                     # matching implementations + main.cc, service_discovery.cc
├── hyperlane_client/            # Python package
│   ├── setup.py, generate_grpc.py
│   ├── hyperlane_client/        # discovery.py, orchestrator.py, grpc_client.py
│   └── pybind/                  # TensorSocket C++ extension + Python wrapper
├── tests.py                     # unit tests (partitioner, discovery)
├── example_inference.py         # end-to-end usage sketch
├── build.sh, setup_env.sh       # build / environment scripts
├── build_wheels_local.sh, test_wheels.sh, upload_wheels.sh   # wheel tooling (not working yet)
└── .github/workflows/           # CI (see Status)
```

## Performance notes

1. **Activation size**: one decode step moves `hidden_size × 2 bytes` per hop (≈8 KB for a 4096-wide model), so decode is latency-bound; long prompts move far more data per hop and are bandwidth-bound.
2. **Quantization**: INT4 shrinks transfers 4× but LLM activations have large outliers; do not enable it without measuring output quality against FP16.
3. **Network**: 1 GbE is workable for decode; use 2.5/10 GbE for long prompts.
4. **Pinned memory**: 512 MB per sender/receiver by default.

## Troubleshooting

```bash
nvidia-smi                          # CUDA visible?
lsof -i :50051                      # port in use?
systemctl status avahi-daemon       # mDNS daemon up?
avahi-browse -a                     # is the worker advertised?
sudo ufw allow 50051:50100/tcp      # firewall
```

## Development

```bash
# Python unit tests (needs the packages in requirements.txt; importing the package pulls in torch)
python3 tests.py -v

# Worker only
cd hyperlane_worker && mkdir -p build && cd build && cmake .. && cmake --build .

# Python client only
cd hyperlane_client && python3 setup.py develop
```

## Roadmap

Path from prototype to a usable product, in order:

- [ ] Build the worker cleanly on Linux and on macOS (Apple Silicon), with a CPU-only build so it can be developed and tested without a GPU
- [ ] Device/backend abstraction so CUDA (NVIDIA) and Metal (Apple Silicon) workers can share one cluster
- [ ] One model, two workers, end to end, with output matching a single-device reference
- [ ] Contiguous, compute- and bandwidth-aware layer partitioner
- [ ] Ship weights to workers (not a client-local file path); authentication/TLS on control and data planes
- [ ] OpenAI-compatible HTTP gateway and a `hyperlane` CLI
- [ ] Health checks, metrics, failure detection and recovery
- [ ] Docker image for Linux/NVIDIA workers; native macOS install (Homebrew + launchd) for Apple Silicon workers; pure-Python client wheel
- [ ] Reproducible benchmarks against single-GPU and existing distributed baselines

Later: TensorRT/other execution backends, dynamic rebalancing, speculative decoding, AMD/Intel GPUs (Vulkan).

## License

No `LICENSE` file has been added yet; MIT is intended.

## Contributing

Pull requests welcome. Use `clang-format` for C++ and `black` for Python, keep I/O paths non-blocking, and validate performance-critical changes on real hardware.
