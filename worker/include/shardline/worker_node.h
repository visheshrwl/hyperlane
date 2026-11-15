#pragma once

#include <memory>
#include <mutex>
#include <string>

#include <grpcpp/grpcpp.h>

#include "shardline/discovery.h"
#include "shardline/worker_service.h"

namespace shardline {

struct WorkerOptions {
  // Loopback by default: the worker has no authentication yet.
  std::string listen_address = "127.0.0.1:50051";
  bool advertise = false;
  std::string worker_id;
};

class WorkerNode {
 public:
  explicit WorkerNode(WorkerOptions options);
  ~WorkerNode();

  WorkerNode(const WorkerNode&) = delete;
  WorkerNode& operator=(const WorkerNode&) = delete;

  bool start();
  void wait();
  void shutdown();

  int port() const { return port_; }
  const std::string& worker_id() const { return worker_id_; }

 private:
  WorkerOptions options_;
  std::string hostname_;
  std::string worker_id_;
  int port_ = 0;

  std::mutex shutdown_mutex_;
  // Declared before server_ so the service outlives the server that uses it.
  std::unique_ptr<WorkerServiceImpl> service_;
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<Advertiser> advertiser_;
};

bool is_loopback_address(const std::string& listen_address);

}  // namespace shardline
