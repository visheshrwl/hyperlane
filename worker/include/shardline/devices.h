#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace shardline {

enum class DeviceKind { kCpu, kMetal, kCuda };

struct DeviceInfo {
  DeviceKind kind = DeviceKind::kCpu;
  std::string name;
  uint64_t total_memory_bytes = 0;
  uint64_t usable_memory_bytes = 0;
};

const char* device_kind_name(DeviceKind kind);

DeviceInfo probe_cpu();
std::vector<DeviceInfo> probe_metal();
std::vector<DeviceInfo> probe_cuda();

// Accelerators first, the host CPU last.
std::vector<DeviceInfo> probe_devices();

}  // namespace shardline
