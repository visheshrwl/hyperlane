#include "shardline/devices.h"

#include <cuda_runtime_api.h>

namespace shardline {

std::vector<DeviceInfo> probe_cuda() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    cudaGetLastError();
    return {};
  }

  std::vector<DeviceInfo> devices;
  for (int index = 0; index < count; ++index) {
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, index) != cudaSuccess ||
        cudaSetDevice(index) != cudaSuccess) {
      cudaGetLastError();
      continue;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
      cudaGetLastError();
      continue;
    }
    DeviceInfo info;
    info.kind = DeviceKind::kCuda;
    info.name = properties.name;
    info.total_memory_bytes = total_bytes;
    info.usable_memory_bytes = free_bytes;
    devices.push_back(std::move(info));
  }
  return devices;
}

}  // namespace shardline
