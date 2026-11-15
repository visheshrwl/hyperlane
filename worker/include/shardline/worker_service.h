#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "shardline/backend.h"
#include "shardline/v1/worker.grpc.pb.h"

namespace shardline {

// The largest gRPC message a worker accepts or sends. The default of 4 MiB
// is smaller than the hidden states of a few hundred tokens of a large model.
inline constexpr int kMaxMessageBytes = 64 << 20;

struct ServiceConfig {
  // Where LoadShard looks for <sha256>.gguf. Empty means no model can load.
  std::string model_dir;
  int threads = 4;
  uint32_t max_sessions = 8;
  // Empty means the ggml engine.
  BackendFactory backend_factory;
};

class WorkerServiceImpl final : public v1::WorkerService::Service {
 public:
  WorkerServiceImpl(std::string worker_id, std::string hostname,
                    ServiceConfig config = {});

  grpc::Status GetStats(grpc::ServerContext* context,
                        const v1::GetStatsRequest* request,
                        v1::GetStatsResponse* response) override;

  grpc::Status LoadShard(grpc::ServerContext* context,
                         const v1::LoadShardRequest* request,
                         v1::LoadShardResponse* response) override;

  grpc::Status Forward(
      grpc::ServerContext* context,
      grpc::ServerReaderWriter<v1::ForwardResponse, v1::ForwardRequest>* stream)
      override;

  grpc::Status CloseSession(grpc::ServerContext* context,
                            const v1::CloseSessionRequest* request,
                            v1::CloseSessionResponse* response) override;

 private:
  // A session and the shard generation it was opened on.
  using Owned = std::pair<std::string, uint64_t>;

  grpc::Status forward_one(const v1::ForwardRequest& request,
                           v1::ForwardResponse* response,
                           std::vector<Owned>* owned);
  void close_owned(const std::vector<Owned>& owned);

  std::string worker_id_;
  std::string hostname_;
  ServiceConfig config_;

  // Guards the shard and every session on it. A backend is not thread safe.
  std::mutex mutex_;
  std::unique_ptr<Backend> backend_;
  uint64_t generation_ = 0;
};

}  // namespace shardline
