#pragma once

#include <grpcpp/grpcpp.h>
#include <memory>
#include "worker.h"
#include "service.grpc.pb.h"

namespace hyperlane::worker {

/**
 * gRPC service implementation for the Worker service.
 * Handles RPC calls from orchestration clients.
 */
class WorkerServiceImpl final : public hyperlane::worker::Worker::Service {
 public:
  explicit WorkerServiceImpl(Worker* worker_instance);

  ::grpc::Status GetStats(::grpc::ServerContext* context,
                         const ::google::protobuf::Empty* request,
                         WorkerStats* response) override;

  ::grpc::Status LoadShard(::grpc::ServerContext* context,
                          const LoadShardRequest* request,
                          LoadShardResponse* response) override;

  ::grpc::Status ExecutePipeline(::grpc::ServerContext* context,
                                 const ExecutionRequest* request,
                                 ExecutionResponse* response) override;

  ::grpc::Status SetNextWorker(::grpc::ServerContext* context,
                              const NextWorkerAddress* request,
                              ::google::protobuf::Empty* response) override;

 private:
  Worker* worker_instance_;
};

}  // namespace hyperlane::worker
