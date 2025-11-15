# Hyperlane

**Distributed GPU Inference Engine for Heterogeneous Prosumer Hardware**

Hyperlane enables single-command LLM inference across multiple consumer GPUs on a local area network. It uses pipeline parallelism to split large models layer-wise and coordinates execution with minimal network overhead.

## Architecture

Hyperlane is split into two core components:

### hyperlane_worker (C++/CUDA)
High-performance inference server with:
- **Control Plane**: gRPC service for model shard loading and orchestration
- **Execution Engine**: ONNX Runtime with CUDA provider for layer execution
- **Data Plane**: Async TCP sockets for inter-worker tensor transmission
- **Service Discovery**: Avahi/mDNS registration with GPU stats in TXT records
- **Optimization**: FP16→INT4 quantization, async CUDA streams, pinned memory DMA

### hyperlane_client (Python)
User-friendly orchestration library with:
- **Discovery**: zeroconf-based auto-discovery of workers on the LAN
- **Sharding**: Knapsack partitioning algorithm for layer distribution
- **ONNX Export**: torch.onnx.export for shard compilation
- **gRPC Orchestration**: Async client for worker control
- **pybind11 Bridge**: Zero-copy tensor socket for efficient inference starts

## Pipeline Parallelism

Models are partitioned **layer-wise** (not tensor-wise) to minimize network latency:

```
Input → GPU A (Layers 1-10) → GPU B (Layers 11-20) → GPU C (Layers 21-32) → Output
```

Each GPU:
1. Executes its layers on the input tensor (FP16/FP32)
2. Quantizes output to INT4 in VRAM
3. Async D2H copy to pinned host memory
4. Sends via TCP socket (DMA-friendly)
5. Next GPU receives and async H2D + dequantizes

This overlap minimizes pipeline bubbles and network blocking.

## Installation

### System Requirements
- Ubuntu 20.04+ (Linux only)
- NVIDIA GPU with CUDA Compute Capability 6.0+
- CUDA Toolkit 11.8+
- cuDNN 8.0+
- gRPC + protobuf
- ONNX Runtime with CUDA provider
- Avahi (mDNS)

### Quick Setup

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake \
  protobuf-compiler libprotobuf-dev \
  libgrpc++-dev protobuf-compiler-grpc \
  libavahi-client-dev libavahi-common-dev \
  python3-dev python3-pip

# Download/install CUDA and cuDNN (if not present)
# https://developer.nvidia.com/cuda-toolkit
# https://developer.nvidia.com/cudnn

# Install onnxruntime-gpu
pip install onnxruntime-gpu

# Clone and build
git clone <repo> hyperlane
cd hyperlane
bash build.sh
```

## Quick Start

### Start a Worker

```bash
# On GPU machine 1 (port 50051 default)
./hyperlane_worker/build/hyperlane_worker

# On GPU machine 2 (port 50052)
./hyperlane_worker/build/hyperlane_worker 50052

# On GPU machine 3 (port 50053)
./hyperlane_worker/build/hyperlane_worker 50053
```

Workers will register themselves via Avahi/mDNS and expose GPU stats.

### Load and Run a Model

```python
import asyncio
from hyperlane_client import DiscoveryManager, AutoDistributedModel

async def main():
    # Discover workers on LAN
    discovery = DiscoveryManager()
    discovery.start_discovery()
    await asyncio.sleep(2)  # Wait for discovery
    
    print(f"Found {len(discovery.discovered_workers)} workers")
    
    # Load model and auto-shard across workers
    model = AutoDistributedModel.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        discovery
    )
    
    # Run inference
    output = model.generate("What is machine learning?", max_tokens=128)
    print(output)
    
    discovery.stop_discovery()

asyncio.run(main())
```

## File Structure

```
.
├── hyperlane_worker/              # C++/CUDA inference server
│   ├── CMakeLists.txt
│   ├── include/                   # Headers
│   │   ├── worker.h              # Main orchestrator
│   │   ├── service_impl.h        # gRPC service
│   │   ├── onnx_session.h        # ONNX Runtime wrapper
│   │   ├── tensor_sender.h       # Async sender
│   │   ├── tensor_receiver.h     # Async receiver
│   │   ├── cuda_ops.h            # Quantization kernels
│   └── src/                       # Implementations
│       ├── main.cc
│       ├── worker.cc
│       ├── service_impl.cc
│       ├── onnx_session.cc
│       ├── tensor_sender.cc
│       ├── tensor_receiver.cc
│       ├── cuda_ops.cu
│       └── service_discovery.cc
├── hyperlane_client/              # Python orchestration
│   ├── setup.py
│   ├── generate_grpc.py          # Proto compilation script
│   ├── hyperlane_client/
│   │   ├── __init__.py
│   │   ├── discovery.py          # Zeroconf discovery
│   │   ├── orchestrator.py       # Model sharding & deployment
│   │   └── grpc_client.py        # Worker gRPC client
│   └── pybind/
│       ├── CMakeLists.txt
│       ├── tensor_socket.h       # pybind11 wrapper
│       ├── tensor_socket.cpp     # Implementation
│       └── tensor_socket.py      # Python interface
├── proto/
│   └── service.proto             # gRPC service definition
├── build.sh                       # Build script
└── README.md                      # This file
```

## Performance Considerations

1. **Quantization**: INT4 reduces tensor size 4x, but verify accuracy on your models.
2. **Network**: 1 Gbps LAN is sufficient for most prosumer setups; higher is better.
3. **Pinned Memory**: Allocate based on available system RAM (512 MB default per worker).
4. **CUDA Streams**: Overlap D2H, socket send, and next H2D for minimal latency.

## Troubleshooting

### Worker doesn't start
```bash
# Check CUDA availability
nvidia-smi

# Check port
lsof -i :50051

# Check Avahi daemon
systemctl status avahi-daemon
```

### Discovery fails
```bash
# Ensure Avahi is running
sudo systemctl start avahi-daemon

# Check mDNS resolution
avahi-browse -a
```

### gRPC connection refused
```bash
# Verify worker is running and listening
netstat -tlnp | grep hyperlane_worker

# Check firewall
sudo ufw allow 50051:50100/tcp
```

## Development

### Build just the worker
```bash
cd hyperlane_worker
mkdir -p build && cd build
cmake .. && cmake --build .
```

### Build just the Python client
```bash
cd hyperlane_client
python3 setup.py develop
```

### Run tests (TODO)
```bash
pytest tests/
```

## Roadmap

- [ ] TensorRT engine support (beyond ONNX)
- [ ] Dynamic layer redistribution based on load
- [ ] Speculative decoding for batch inference
- [ ] Quantization-aware training (QAT)
- [ ] Web dashboard for monitoring
- [ ] Support for other accelerators (AMD, Intel)

## License

MIT (or chosen license)

## Contributing

Pull requests welcome. Please ensure:
- Code follows project style (clang-format for C++, black for Python)
- All async code is non-blocking
- Performance-critical paths are validated on real hardware

