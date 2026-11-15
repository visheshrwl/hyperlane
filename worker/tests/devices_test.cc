#include <cstdlib>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "shardline/devices.h"

namespace shardline {
namespace {

TEST(Devices, CpuProbeReportsMemory) {
  DeviceInfo cpu = probe_cpu();
  EXPECT_EQ(cpu.kind, DeviceKind::kCpu);
  EXPECT_FALSE(cpu.name.empty());
  EXPECT_GT(cpu.total_memory_bytes, 0u);
  EXPECT_GT(cpu.usable_memory_bytes, 0u);
  EXPECT_LE(cpu.usable_memory_bytes, cpu.total_memory_bytes);
}

TEST(Devices, CpuIsListedLast) {
  std::vector<DeviceInfo> devices = probe_devices();
  ASSERT_FALSE(devices.empty());
  EXPECT_EQ(devices.back().kind, DeviceKind::kCpu);
  for (size_t i = 0; i + 1 < devices.size(); ++i) {
    EXPECT_NE(devices[i].kind, DeviceKind::kCpu);
  }
}

TEST(Devices, AcceleratorMemoryIsConsistent) {
  for (const DeviceInfo& device : probe_devices()) {
    EXPECT_GT(device.total_memory_bytes, 0u) << device.name;
    EXPECT_LE(device.usable_memory_bytes, device.total_memory_bytes)
        << device.name;
  }
}

#if defined(__APPLE__)
// Hosted CI runners are virtual machines and may expose no Metal device, so
// the device is only required when SHARDLINE_EXPECT_METAL=1.
TEST(Devices, MetalWorkingSetIsBelowSystemMemory) {
  std::vector<DeviceInfo> metal = probe_metal();
  const char* expect = std::getenv("SHARDLINE_EXPECT_METAL");
  if (metal.empty()) {
    if (expect != nullptr && std::string(expect) == "1") {
      FAIL() << "no Metal device found";
    }
    GTEST_SKIP() << "no Metal device on this machine";
  }
  ASSERT_EQ(metal.size(), 1u);
  std::cout << "Metal device: " << metal[0].name
            << ", usable bytes: " << metal[0].usable_memory_bytes
            << ", total bytes: " << metal[0].total_memory_bytes << "\n";
  EXPECT_LT(metal[0].usable_memory_bytes, metal[0].total_memory_bytes);
}
#endif

}  // namespace
}  // namespace shardline
