# Hyperlane Development Guide

This guide provides detailed information for developers working on the Hyperlane project.

## Architecture Overview

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Python Client (hyperlane_client)                            │
├─────────────────────────────────────────────────────────────┤
│ 1. Discovery: Zeroconf mDNS for worker discovery            │
│ 2. Sharding: KnapsackPartitioner splits model layers        │
│ 3. ONNX Export: torch.onnx.export for each shard            │
│ 4. gRPC Orchestration: LoadShard RPCs to workers            │
│ 5. TensorSocket (pybind11): Zero-copy tensor transmission   │
└────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ C++ Worker (hyperlane_worker)                               │
├─────────────────────────────────────────────────────────────┤
│ Control Plane:                                              │
│  • gRPC Worker service (GetStats, LoadShard, Execute)       │
│  • Avahi/mDNS service registration                          │
│                                                              │
│ Execution Engine:                                           │
│  • ONNX Runtime with CUDA provider                          │
│  • Model shard loading and inference                        │
│                                                              │
│ Data Plane:                                                 │
│  • TensorSender: Async D2H + socket send                    │
│  • TensorReceiver: Async socket recv + H2D                  │
│  • CUDA Ops: FP16↔INT4 quantization kernels                │
└────────────────────────────────────────────────────────────┘
```

## Key Components

### Python Client Components

#### `discovery.py`
- **DiscoveryManager**: Uses zeroconf (mDNS) to auto-discover `_hyperlane._tcp` services
- Parses TXT records for GPU stats (name, VRAM)
- Non-blocking async service browser

#### `orchestrator.py`
- **KnapsackPartitioner**: Greedy bin-packing for layer distribution
- **AutoDistributedModel**: Main orchestration API
  - `from_pretrained()`: Load HF model, shard, and deploy
  - `_extract_transformer_layers()`: Model architecture agnostic
  - `_export_shards()`: torch.onnx.export for each partition
  - `_deploy_shards()`: gRPC LoadShard calls
  - `generate()`: End-to-end inference

#### `grpc_client.py`
- **WorkerClient**: Async gRPC stub for single worker
- **WorkerPool**: Manages multiple worker connections
- Stubs for: GetStats, LoadShard, ExecutePipeline, SetNextWorker

#### `pybind/tensor_socket.py` (Python wrapper)
- Imports C++ extension `hyperlane_tensor_socket`
- Exposes **TensorSocket** class for zero-copy tensor transmission
- Fallback pure-Python stub if C++ extension not built

### C++ Worker Components

#### `worker.h / worker.cc`
- **Worker**: Main orchestrator
  - `initialize()`: GPU detection, Avahi registration, gRPC startup
  - `run()`: Event loop (gRPC server blocks here)
  - `shutdown()`: Cleanup
- **GPUStats**: GPU metadata (name, VRAM, formats)
- **ExecutionEngine**: Loads and runs ONNX shards

#### `service_impl.h / service_impl.cc`
- **WorkerServiceImpl**: gRPC service implementation
- RPC handlers:
  - `GetStats()`: Return worker GPU info
  - `LoadShard()`: Load ONNX model
  - `ExecutePipeline()`: Trigger inference (placeholder)
  - `SetNextWorker()`: Configure next pipeline stage

#### `onnx_session.h / onnx_session.cc`
- **ONNXSessionWrapper**: Wraps Ort::Session
- `initialize()`: Creates session with CUDA provider
- `run()`: Executes inference with GPU memory binding

#### `tensor_sender.h / tensor_sender.cc`
- **TensorSender**: Async inter-worker transmission
- Manages:
  - Pinned host memory allocation (512 MB default)
  - CUDA streams for D2H overlap
  - TCP socket (async connection)
  - Handle-based send queue
  - Background send thread
- Key methods:
  - `async_send_tensor()`: Quantize, D2H, queue send
  - `is_send_complete()`, `wait_send_complete()`: Handle polling

#### `tensor_receiver.h / tensor_receiver.cc`
- **TensorReceiver**: Async inter-worker reception
- Manages:
  - Pinned host memory allocation
  - CUDA streams for H2D overlap
  - TCP listen socket
  - Background recv thread
- Key methods:
  - `start_listening()`: Accept incoming connections
  - `async_recv_tensor()`: Queue recv, H2D, dequantize

#### `cuda_ops.cu`
- **quantize_fp16_to_int4_kernel**: VRAM-resident FP16→INT4
  - Packs 4 FP16 values into 2 bytes
  - Reduces tensor size 4x
- **dequantize_int4_to_fp16_kernel**: INT4→FP16
- Async wrappers on CUDA streams

#### `service_discovery.cc`
- **ServiceDiscovery**: Avahi/mDNS registration
- `publish_service()`: Register `_hyperlane._tcp` with GPU stats in TXT record
- `unpublish_service()`: Deregister on shutdown

#### `pybind/tensor_socket.h / tensor_socket.cpp`
- **PyTensorSocket**: C++ pybind11 wrapper
- Exposes to Python:
  - `connect()`: Establish socket
  - `async_send()`: Send NumPy/PyTorch array (zero-copy via `py::array_t<>`)
  - `is_complete()`, `wait_complete()`: Handle polling
- PYBIND11_MODULE: Bindings for Python import

## Build System

### CMakeLists.txt (hyperlane_worker)
- Finds: CUDA, gRPC, protobuf, onnxruntime, avahi
- Compiles protobuf/gRPC stubs
- Builds worker executable with CUDA architecture flags
- Links: libgrpc++, libprotobuf, libonnxruntime, libcuda, libavahi-client

### CMakeLists.txt (hyperlane_client/pybind)
- Finds: Python3, pybind11
- Builds shared library (.so) with pybind11
- Output: `hyperlane_tensor_socket.so`

### build.sh
- Master build script
- Checks dependencies
- Generates gRPC Python stubs (protoc)
- Builds C++ worker
- Builds pybind11 extension
- Installs Python package

## Development Workflow

### Adding a New RPC Endpoint

1. **Update proto** (`proto/service.proto`)
   ```protobuf
   rpc NewEndpoint(NewRequest) returns (NewResponse);
   ```

2. **Regenerate stubs**
   ```bash
   python3 hyperlane_client/generate_grpc.py
   ```

3. **Implement handler** (`hyperlane_worker/src/service_impl.cc`)
   ```cpp
   ::grpc::Status WorkerServiceImpl::NewEndpoint(...) { ... }
   ```

4. **Rebuild worker**
   ```bash
   cd hyperlane_worker/build && cmake --build .
   ```

### Adding a New CUDA Kernel

1. **Declare** in `include/cuda_ops.h`
2. **Implement** in `src/cuda_ops.cu` (use `__global__` + wrapper)
3. **Rebuild worker** (CMake auto-detects .cu files)

### Debugging

#### Worker Logging
```cpp
std::cout << "[Component] Message\n";
std::cerr << "[Error] Message\n";
```

#### Python Client Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Message")
```

#### Avahi Service Registration
```bash
avahi-browse -a  # List all mDNS services
avahi-resolve-host-name <hostname>.local  # Resolve mDNS hostname
```

#### gRPC Debugging
```python
import grpc
grpc.aio.secure_channel(..., options=[
    ("grpc.max_receive_message_length", -1),
])
```

### Testing

Currently minimal test coverage. To add:

```bash
pytest tests/
```

Suggested test areas:
- Quantization accuracy (INT4 vs FP16)
- ONNX export correctness
- Discovery timeout handling
- Socket error recovery

## Performance Profiling

### Measure CUDA Kernel Time
```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
quantize_fp16_to_int4_kernel<<<grid, block, 0, stream>>>(input, output, n);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
std::cout << "Kernel time: " << ms << " ms\n";
```

### Profile gRPC Latency
```python
import time

start = time.perf_counter()
response = await client.stub.GetStats(...)
elapsed = (time.perf_counter() - start) * 1000
print(f"RPC latency: {elapsed:.2f} ms")
```

### Measure Network Throughput
Use `iperf3` between machines:
```bash
# On receiver
iperf3 -s

# On sender
iperf3 -c <receiver_ip>
```

## Common Issues

### CUDA Out of Memory
- Reduce pinned buffer size in `tensor_sender.h` / `tensor_receiver.h`
- Check model size vs available VRAM

### gRPC Port Already in Use
```bash
lsof -i :50051
kill -9 <PID>
```

### Avahi Service Not Visible
```bash
# Ensure daemon is running
sudo systemctl restart avahi-daemon

# Check firewall
sudo ufw allow mdns
```

### ONNX Export Fails
- Verify model architecture is in supported list
- Check torch version compatibility
- Use `verbose=True` in export call

## Useful References

- [gRPC C++ API](https://grpc.io/docs/languages/cpp/basics/)
- [ONNX Runtime C++ API](https://onnx.ai/onnx/repo-docs/c-ort-env.html)
- [pybind11 docs](https://pybind11.readthedocs.io/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [Avahi C API](https://avahi.org/doxygen/html/index.html)
