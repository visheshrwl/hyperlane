#include "shardline/devices.h"

#import <Metal/Metal.h>
#include <sys/sysctl.h>

namespace shardline {

std::vector<DeviceInfo> probe_metal() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      return {};
    }

    uint64_t ram = 0;
    size_t size = sizeof(ram);
    sysctlbyname("hw.memsize", &ram, &size, nullptr, 0);

    uint64_t working_set = device.recommendedMaxWorkingSetSize;
    uint64_t allocated = device.currentAllocatedSize;

    DeviceInfo info;
    info.kind = DeviceKind::kMetal;
    info.name = device.name.UTF8String;
    info.total_memory_bytes = device.hasUnifiedMemory ? ram : working_set;
    info.usable_memory_bytes = metal_usable_memory(
        working_set, allocated, host_available_memory(), device.hasUnifiedMemory);
    return {info};
  }
}

}  // namespace shardline
