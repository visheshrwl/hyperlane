#include "worker.h"
#include "service_impl.h"
#include <iostream>
#include <unistd.h>
#include <grpcpp/grpcpp.h>
#include <memory>

namespace hyperlane::worker {

Worker::Worker(uint16_t grpc_port)
    : grpc_port_(grpc_port),
      execution_engine_(std::make_unique<ExecutionEngine>()),
      gpu_stats_(std::make_unique<GPUStats>()),
      grpc_server_(nullptr) {
}

Worker::~Worker() = default;

bool Worker::initialize() {
  // Detect GPU info
  gpu_stats_->worker_id = "worker_" + std::to_string(getpid());
  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  gpu_stats_->hostname = hostname;
  // TODO: Query actual GPU via CUDA API
  gpu_stats_->gpu_name = "NVIDIA_GPU";
  gpu_stats_->total_memory = 24UL * 1024 * 1024 * 1024;  // 24GB placeholder
  gpu_stats_->free_memory = gpu_stats_->total_memory;
  gpu_stats_->supported_formats = {"onnx", "trt"};

  // Initialize service discovery
  service_discovery_ = std::make_unique<ServiceDiscovery>(
      gpu_stats_->hostname, grpc_port_, *gpu_stats_);
  if (!service_discovery_->publish_service()) {
    std::cerr << "Failed to publish service via Avahi\n";
    return false;
  }

  // Initialize gRPC server
  std::string server_address = "0.0.0.0:" + std::to_string(grpc_port_);
  WorkerServiceImpl service(this);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  grpc_server_ = builder.BuildAndStart();
  if (!grpc_server_) {
    std::cerr << "Failed to start gRPC server\n";
    return false;
  }

  std::cout << "gRPC server listening on " << server_address << "\n";

  return true;
}

void Worker::run() {
  std::cout << "Worker started on port " << grpc_port_ << "\n";
  if (grpc_server_) {
    grpc_server_->Wait();
  }
}

void Worker::shutdown() {
  if (grpc_server_) {
    grpc_server_->Shutdown();
  }
  if (service_discovery_) {
    service_discovery_->unpublish_service();
  }
}

GPUStats Worker::get_stats() const {
  return *gpu_stats_;
}

bool Worker::load_shard(const std::string& shard_name,
                        const std::string& model_path) {
  return execution_engine_->load_onnx_shard(model_path, shard_name);
}

ExecutionEngine::ExecutionEngine() = default;
ExecutionEngine::~ExecutionEngine() = default;

bool ExecutionEngine::load_onnx_shard(const std::string& onnx_path,
                                      const std::string& shard_name) {
  std::cout << "Loading ONNX shard: " << shard_name << " from " << onnx_path
            << "\n";
  
  auto session_wrapper = std::make_unique<ONNXSessionWrapper>(onnx_path);
  if (!session_wrapper->initialize()) {
    return false;
  }

  sessions_[shard_name] = std::move(session_wrapper);
  return true;
}

bool ExecutionEngine::run_shard(const std::string& shard_name,
                                const void* input_gpu_buffer,
                                void** output_gpu_buffer_out) {
  std::cout << "Running shard: " << shard_name << "\n";
  
  auto it = sessions_.find(shard_name);
  if (it == sessions_.end()) {
    std::cerr << "Shard not found: " << shard_name << "\n";
    return false;
  }

  // TODO: Run the ONNX session
  return true;
}

TensorSender::TensorSender(const std::string& next_worker_address,
                           uint16_t next_port)
    : next_worker_address_(next_worker_address),
      next_port_(next_port),
      socket_fd_(-1),
      pinned_buffer_(nullptr),
      pinned_buffer_size_(0) {
}

TensorSender::~TensorSender() {
  // TODO: Clean up socket and pinned memory
}

int TensorSender::async_send_tensor(const void* device_ptr,
                                     const std::vector<int64_t>& shape,
                                     const std::string& tensor_name) {
  std::cout << "Async send tensor: " << tensor_name << "\n";
  // TODO: Quantize, D2H, send
  return 0;
}

bool TensorSender::is_send_complete(int handle) {
  // TODO: Check handle status
  return true;
}

void TensorSender::wait_send_complete(int handle) {
  // TODO: Block until complete
}

TensorReceiver::TensorReceiver(uint16_t listen_port)
    : listen_port_(listen_port),
      listen_socket_fd_(-1),
      pinned_recv_buffer_(nullptr),
      pinned_recv_buffer_size_(0) {
}

TensorReceiver::~TensorReceiver() {
  // TODO: Clean up listen socket and pinned memory
}

void TensorReceiver::start_listening() {
  std::cout << "Starting tensor receiver on port " << listen_port_ << "\n";
  // TODO: Create and bind listen socket
}

int TensorReceiver::async_recv_tensor(void** device_ptr_out) {
  std::cout << "Async recv tensor\n";
  // TODO: Recv, H2D, dequantize
  return 0;
}

bool TensorReceiver::is_recv_complete(int handle) {
  // TODO: Check handle status
  return true;
}

void TensorReceiver::wait_recv_complete(int handle) {
  // TODO: Block until complete
}

ServiceDiscovery::ServiceDiscovery(const std::string& hostname,
                                   uint16_t grpc_port, const GPUStats& stats)
    : hostname_(hostname), grpc_port_(grpc_port), gpu_stats_(stats) {
}

ServiceDiscovery::~ServiceDiscovery() = default;

bool ServiceDiscovery::publish_service() {
  std::cout << "Publishing service via Avahi: " << hostname_ << ":"
            << grpc_port_ << "\n";
  // TODO: Register _hyperlane._tcp service with GPU stats in TXT record
  return true;
}

bool ServiceDiscovery::unpublish_service() {
  std::cout << "Unpublishing service\n";
  // TODO: Deregister service
  return true;
}

}  // namespace hyperlane::worker
