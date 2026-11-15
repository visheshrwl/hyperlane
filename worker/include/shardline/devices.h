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

// Host RAM that is free or reclaimable right now; 0 when it cannot be read.
uint64_t host_available_memory();

// What a Metal device may use. The recommended working set is a static limit
// that ignores what other processes hold, so on unified memory the result is
// also capped by the RAM the host has available. A host figure of 0 means
// unknown and leaves the working-set headroom as is.
uint64_t metal_usable_memory(uint64_t working_set, uint64_t allocated,
                             uint64_t host_available, bool unified_memory);

}  // namespace shardline
