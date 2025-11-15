#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <map>
#include <grpcpp/grpcpp.h>

namespace hyperlane::worker {

// Forward declaration
struct TensorBuffer;

/**
 * Represents statistics about the worker's GPU.
 */
struct GPUStats {
  std::string worker_id;
  std::string hostname;
  std::string gpu_name;
  uint64_t total_memory = 0;
  uint64_t free_memory = 0;
  std::vector<std::string> supported_formats;
};

/**
 * Asynchronous tensor sender for pipeline stage-to-stage communication.
 * Handles:
 *  - FP16/FP32 -> INT4 quantization in VRAM (custom CUDA kernel)
 *  - Async D2H copy to pinned host buffer
 *  - Non-blocking TCP socket send with NIC DMA
 */
class TensorSender;

/**
 * Asynchronous tensor receiver for receiving tensors from previous pipeline stage.
 * Handles:
 *  - Non-blocking TCP socket recv into pinned buffer
 *  - Async H2D copy from pinned to GPU VRAM
 *  - INT4 -> FP16/FP32 dequantization in VRAM (custom CUDA kernel)
 */
class TensorReceiver;

/**
 * Execution engine: loads ONNX models and runs inference.
 */
class ExecutionEngine {
 public:
  ExecutionEngine();
  ~ExecutionEngine();

  // Load an ONNX model shard
  bool load_onnx_shard(const std::string& onnx_path, const std::string& shard_name);

  // Run inference on a shard
  bool run_shard(const std::string& shard_name, const void* input_gpu_buffer,
                 void** output_gpu_buffer_out);

 private:
  std::map<std::string, std::unique_ptr<class ONNXSessionWrapper>> sessions_;
};

// Forward declare
class ONNXSessionWrapper;

/**
 * Service discovery via Avahi/mDNS.
 */
class ServiceDiscovery {
 public:
  ServiceDiscovery(const std::string& hostname, uint16_t grpc_port,
                   const GPUStats& stats);
  ~ServiceDiscovery();

  bool publish_service();
  bool unpublish_service();

 private:
  std::string hostname_;
  uint16_t grpc_port_;
  GPUStats gpu_stats_;
  class AvahiContext* avahi_context_;
};

/**
 * Main worker orchestrator.
 */
class Worker {
 public:
  explicit Worker(uint16_t grpc_port);
  ~Worker();

  bool initialize();
  void run();
  void shutdown();

  GPUStats get_stats() const;
  bool load_shard(const std::string& shard_name, const std::string& model_path);

 private:
  uint16_t grpc_port_;
  std::unique_ptr<ExecutionEngine> execution_engine_;
  std::unique_ptr<TensorSender> tensor_sender_;
  std::unique_ptr<TensorReceiver> tensor_receiver_;
  std::unique_ptr<ServiceDiscovery> service_discovery_;
  std::unique_ptr<GPUStats> gpu_stats_;
  std::unique_ptr<grpc::Server> grpc_server_;
};

}  // namespace hyperlane::worker
