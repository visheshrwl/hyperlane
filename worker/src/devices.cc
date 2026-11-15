#include "shardline/devices.h"

#include <unistd.h>

#include <algorithm>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#else
#include <fstream>
#include <sstream>
#endif

namespace shardline {

const char* device_kind_name(DeviceKind kind) {
  switch (kind) {
    case DeviceKind::kCpu:
      return "cpu";
    case DeviceKind::kMetal:
      return "metal";
    case DeviceKind::kCuda:
      return "cuda";
  }
  return "unknown";
}

namespace {

#if defined(__APPLE__)

std::string cpu_name() {
  char buffer[256] = {};
  size_t size = sizeof(buffer) - 1;
  if (sysctlbyname("machdep.cpu.brand_string", buffer, &size, nullptr, 0) != 0) {
    return "CPU";
  }
  return buffer;
}

uint64_t total_ram() {
  uint64_t bytes = 0;
  size_t size = sizeof(bytes);
  if (sysctlbyname("hw.memsize", &bytes, &size, nullptr, 0) != 0) {
    return 0;
  }
  return bytes;
}

// macOS has no "available memory" counter; free + inactive pages is the usual
// approximation. free_count already includes the speculative pages (vm_stat
// prints free_count - speculative_count), so they must not be added again.
uint64_t available_ram() {
  vm_statistics64_data_t stats{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                        reinterpret_cast<host_info64_t>(&stats),
                        &count) != KERN_SUCCESS) {
    return 0;
  }
  vm_size_t page_size = 0;
  host_page_size(mach_host_self(), &page_size);
  uint64_t pages = static_cast<uint64_t>(stats.free_count) + stats.inactive_count;
  return pages * page_size;
}

#else

uint64_t meminfo_bytes(const std::string& key) {
  std::ifstream meminfo("/proc/meminfo");
  std::string line;
  while (std::getline(meminfo, line)) {
    if (line.rfind(key + ":", 0) == 0) {
      std::istringstream fields(line.substr(key.size() + 1));
      uint64_t kib = 0;
      fields >> kib;
      return kib * 1024;
    }
  }
  return 0;
}

std::string cpu_name() {
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    if (line.rfind("model name", 0) == 0) {
      auto colon = line.find(':');
      if (colon != std::string::npos && colon + 2 <= line.size()) {
        return line.substr(colon + 2);
      }
    }
  }
  return "CPU";
}

uint64_t total_ram() {
  uint64_t bytes = meminfo_bytes("MemTotal");
  if (bytes == 0) {
    bytes = static_cast<uint64_t>(sysconf(_SC_PHYS_PAGES)) *
            static_cast<uint64_t>(sysconf(_SC_PAGE_SIZE));
  }
  return bytes;
}

uint64_t available_ram() { return meminfo_bytes("MemAvailable"); }

#endif

}  // namespace

uint64_t host_available_memory() { return available_ram(); }

uint64_t metal_usable_memory(uint64_t working_set, uint64_t allocated,
                             uint64_t host_available, bool unified_memory) {
  uint64_t headroom = working_set > allocated ? working_set - allocated : 0;
  if (unified_memory && host_available != 0) {
    return std::min(headroom, host_available);
  }
  return headroom;
}

DeviceInfo probe_cpu() {
  DeviceInfo info;
  info.kind = DeviceKind::kCpu;
  info.name = cpu_name();
  info.total_memory_bytes = total_ram();
  info.usable_memory_bytes = std::min(available_ram(), info.total_memory_bytes);
  return info;
}

#if !defined(SHARDLINE_HAVE_METAL)
std::vector<DeviceInfo> probe_metal() { return {}; }
#endif

#if !defined(SHARDLINE_HAVE_CUDA)
std::vector<DeviceInfo> probe_cuda() { return {}; }
#endif

std::vector<DeviceInfo> probe_devices() {
  std::vector<DeviceInfo> devices = probe_cuda();
  for (auto& device : probe_metal()) {
    devices.push_back(std::move(device));
  }
  devices.push_back(probe_cpu());
  return devices;
}

}  // namespace shardline
