#include "shardline/worker_service.h"

#include "shardline/devices.h"
#include "shardline/protocol.h"

namespace shardline {

namespace {

v1::DeviceKind to_proto(DeviceKind kind) {
  switch (kind) {
    case DeviceKind::kCpu:
      return v1::DEVICE_KIND_CPU;
    case DeviceKind::kMetal:
      return v1::DEVICE_KIND_METAL;
    case DeviceKind::kCuda:
      return v1::DEVICE_KIND_CUDA;
  }
  return v1::DEVICE_KIND_UNSPECIFIED;
}

}  // namespace

const char* worker_version() { return SHARDLINE_VERSION; }

WorkerServiceImpl::WorkerServiceImpl(std::string worker_id,
                                     std::string hostname)
    : worker_id_(std::move(worker_id)), hostname_(std::move(hostname)) {}

grpc::Status WorkerServiceImpl::GetStats(grpc::ServerContext*,
                                         const v1::GetStatsRequest*,
                                         v1::GetStatsResponse* response) {
  response->set_worker_id(worker_id_);
  response->set_hostname(hostname_);
  response->set_protocol(kProtocol);
  response->set_worker_version(worker_version());
  for (const DeviceInfo& device : probe_devices()) {
    v1::DeviceInfo* entry = response->add_devices();
    entry->set_kind(to_proto(device.kind));
    entry->set_name(device.name);
    entry->set_total_memory_bytes(device.total_memory_bytes);
    entry->set_usable_memory_bytes(device.usable_memory_bytes);
  }
  return grpc::Status::OK;
}

}  // namespace shardline
