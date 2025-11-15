#include "service_impl.h"
#include <iostream>

namespace hyperlane::worker {

WorkerServiceImpl::WorkerServiceImpl(Worker* worker_instance)
    : worker_instance_(worker_instance) {
}

::grpc::Status WorkerServiceImpl::GetStats(::grpc::ServerContext* context,
                                         const ::google::protobuf::Empty* request,
                                         WorkerStats* response) {
  (void)context;
  (void)request;

  auto stats = worker_instance_->get_stats();

  response->set_worker_id(stats.worker_id);
  response->set_hostname(stats.hostname);
  response->set_gpu_name(stats.gpu_name);
  response->set_total_memory(stats.total_memory);
  response->set_free_memory(stats.free_memory);

  for (const auto& fmt : stats.supported_formats) {
    response->add_supported_formats(fmt);
  }

  return ::grpc::Status::OK;
}

::grpc::Status WorkerServiceImpl::LoadShard(::grpc::ServerContext* context,
                                          const LoadShardRequest* request,
                                          LoadShardResponse* response) {
  (void)context;

  std::cout << "[gRPC] LoadShard request: " << request->shard_name() << " from "
            << request->model_path() << "\n";

  bool success = worker_instance_->load_shard(request->shard_name(),
                                              request->model_path());

  response->set_success(success);
  response->set_message(success ? "Shard loaded" : "Failed to load shard");

  // Store next worker address for tensor routing
  if (!request->next_worker_address().empty()) {
    // TODO: Extract address and port, configure TensorSender
  }

  return ::grpc::Status::OK;
}

::grpc::Status WorkerServiceImpl::ExecutePipeline(
    ::grpc::ServerContext* context, const ExecutionRequest* request,
    ExecutionResponse* response) {
  (void)context;

  std::cout << "[gRPC] ExecutePipeline request: " << request->run_id() << "\n";

  response->set_accepted(true);
  response->set_status("Execution queued");

  // TODO: Queue execution request, return handle
  // Actual execution happens via socket data plane

  return ::grpc::Status::OK;
}

::grpc::Status WorkerServiceImpl::SetNextWorker(
    ::grpc::ServerContext* context, const NextWorkerAddress* request,
    ::google::protobuf::Empty* response) {
  (void)context;
  (void)response;

  std::cout << "[gRPC] SetNextWorker: " << request->address() << "\n";

  // TODO: Parse address and port, update TensorSender endpoint

  return ::grpc::Status::OK;
}

}  // namespace hyperlane::worker
