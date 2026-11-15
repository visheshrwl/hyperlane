#include "shardline/worker_node.h"

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>

#include "shardline/devices.h"

namespace shardline {

namespace {

std::string local_hostname() {
  char buffer[256] = {};
  if (gethostname(buffer, sizeof(buffer) - 1) != 0) {
    return "unknown";
  }
  return buffer;
}

std::string random_suffix() {
  std::random_device device;
  std::uniform_int_distribution<uint32_t> distribution;
  char buffer[9];
  std::snprintf(buffer, sizeof(buffer), "%08x", distribution(device));
  return buffer;
}

std::string host_part(const std::string& listen_address) {
  if (!listen_address.empty() && listen_address.front() == '[') {
    auto end = listen_address.find(']');
    return end == std::string::npos ? listen_address
                                    : listen_address.substr(1, end - 1);
  }
  auto colon = listen_address.rfind(':');
  return colon == std::string::npos ? listen_address
                                    : listen_address.substr(0, colon);
}

}  // namespace

bool is_loopback_address(const std::string& listen_address) {
  std::string host = host_part(listen_address);
  return host == "localhost" || host == "::1" || host.rfind("127.", 0) == 0;
}

WorkerNode::WorkerNode(WorkerOptions options)
    : options_(std::move(options)), hostname_(local_hostname()) {
  worker_id_ = options_.worker_id.empty()
                   ? hostname_ + "-" + random_suffix()
                   : options_.worker_id;
}

WorkerNode::~WorkerNode() { shutdown(); }

bool WorkerNode::start() {
  service_ = std::make_unique<WorkerServiceImpl>(worker_id_, hostname_);

  grpc::ServerBuilder builder;
  builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
  builder.AddListeningPort(options_.listen_address,
                           grpc::InsecureServerCredentials(), &port_);
  builder.RegisterService(service_.get());
  server_ = builder.BuildAndStart();
  if (!server_ || port_ == 0) {
    std::cerr << "cannot listen on " << options_.listen_address << "\n";
    server_.reset();
    service_.reset();
    port_ = 0;
    return false;
  }

  if (!is_loopback_address(options_.listen_address)) {
    std::cerr << "WARNING: listening on " << options_.listen_address
              << " without authentication; any host that can reach this "
                 "port can control the worker\n";
  }

  if (options_.advertise) {
    if (is_loopback_address(options_.listen_address)) {
      std::cerr << "mDNS: not advertising a loopback-only listener\n";
    } else if (!(advertiser_ = make_mdns_advertiser())) {
      std::cerr << "mDNS: this build has no mDNS support\n";
    } else if (!advertiser_->start(worker_id_, static_cast<uint16_t>(port_),
                                   build_txt_record(worker_id_,
                                                    probe_devices()))) {
      advertiser_.reset();
    }
  }
  return true;
}

void WorkerNode::wait() {
  if (server_) {
    server_->Wait();
  }
}

void WorkerNode::shutdown() {
  std::lock_guard<std::mutex> lock(shutdown_mutex_);
  if (advertiser_) {
    advertiser_->stop();
    advertiser_.reset();
  }
  if (server_) {
    server_->Shutdown(std::chrono::system_clock::now() +
                      std::chrono::seconds(5));
  }
}

}  // namespace shardline
