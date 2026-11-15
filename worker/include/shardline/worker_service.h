#pragma once

#include <string>

#include "shardline/v1/worker.grpc.pb.h"

namespace shardline {

class WorkerServiceImpl final : public v1::WorkerService::Service {
 public:
  WorkerServiceImpl(std::string worker_id, std::string hostname);

  grpc::Status GetStats(grpc::ServerContext* context,
                        const v1::GetStatsRequest* request,
                        v1::GetStatsResponse* response) override;

 private:
  std::string worker_id_;
  std::string hostname_;
};

}  // namespace shardline
