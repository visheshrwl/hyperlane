#include <arpa/inet.h>
#include <dns_sd.h>
#include <sys/select.h>

#include <chrono>
#include <cstdlib>
#include <map>
#include <random>
#include <string>

#include <gtest/gtest.h>

#include "shardline/discovery.h"

namespace shardline {
namespace {

struct Resolved {
  bool done = false;
  uint16_t port = 0;
  std::map<std::string, std::string> txt;
};

void DNSSD_API on_resolved(DNSServiceRef, DNSServiceFlags, uint32_t,
                           DNSServiceErrorType error, const char*,
                           const char*, uint16_t port, uint16_t txt_length,
                           const unsigned char* txt_bytes, void* context) {
  auto* resolved = static_cast<Resolved*>(context);
  if (error != kDNSServiceErr_NoError) {
    return;
  }
  resolved->port = ntohs(port);
  uint16_t count = TXTRecordGetCount(txt_length, txt_bytes);
  for (uint16_t i = 0; i < count; ++i) {
    char key[256] = {};
    uint8_t value_length = 0;
    const void* value = nullptr;
    if (TXTRecordGetItemAtIndex(txt_length, txt_bytes, i, sizeof(key), key,
                                &value_length, &value) == kDNSServiceErr_NoError) {
      resolved->txt[key] =
          std::string(static_cast<const char*>(value), value_length);
    }
  }
  resolved->done = true;
}

bool mdns_required() {
  const char* value = std::getenv("SHARDLINE_REQUIRE_MDNS");
  return value != nullptr && std::string(value) == "1";
}

// Registers through the system mDNS daemon and resolves the record back, so
// it needs mDNSResponder (macOS) or avahi-daemon (Linux). It is skipped when
// no daemon is reachable unless SHARDLINE_REQUIRE_MDNS=1.
TEST(DnsSdAdvertiser, PublishesOneTxtKeyPerField) {
  std::random_device random;
  std::string name = "shardline-test-" + std::to_string(random());
  std::vector<DeviceInfo> devices = {
      {DeviceKind::kCuda, "NVIDIA GeForce RTX 3090", 25769803776ull,
       24000000000ull}};

  auto advertiser = make_mdns_advertiser();
  ASSERT_NE(advertiser, nullptr);
  if (!advertiser->start(name, 45678, build_txt_record(name, devices))) {
    if (mdns_required()) {
      FAIL() << "mDNS daemon unavailable";
    }
    GTEST_SKIP() << "mDNS daemon unavailable";
  }

  Resolved resolved;
  DNSServiceRef ref = nullptr;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!resolved.done && std::chrono::steady_clock::now() < deadline) {
    if (ref == nullptr &&
        DNSServiceResolve(&ref, 0, kDNSServiceInterfaceIndexAny, name.c_str(),
                          kServiceType, "local.", on_resolved,
                          &resolved) != kDNSServiceErr_NoError) {
      ref = nullptr;
      continue;
    }
    int fd = DNSServiceRefSockFD(ref);
    fd_set readable;
    FD_ZERO(&readable);
    FD_SET(fd, &readable);
    timeval timeout{0, 200000};
    if (select(fd + 1, &readable, nullptr, nullptr, &timeout) > 0) {
      DNSServiceProcessResult(ref);
    }
  }
  if (ref != nullptr) {
    DNSServiceRefDeallocate(ref);
  }
  advertiser->stop();

  ASSERT_TRUE(resolved.done) << "service was not resolved within 10 s";
  EXPECT_EQ(resolved.port, 45678);
  EXPECT_EQ(resolved.txt["proto"], "shardline.v1");
  EXPECT_EQ(resolved.txt["id"], name);
  EXPECT_EQ(resolved.txt["kind"], "cuda");
  EXPECT_EQ(resolved.txt["device"], "NVIDIA GeForce RTX 3090");
  EXPECT_EQ(resolved.txt["mem_total"], "25769803776");
  EXPECT_EQ(resolved.txt["mem_usable"], "24000000000");
}

}  // namespace
}  // namespace shardline
