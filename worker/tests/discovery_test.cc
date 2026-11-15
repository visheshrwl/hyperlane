#include <map>

#include <gtest/gtest.h>

#include "shardline/discovery.h"

namespace shardline {
namespace {

std::map<std::string, std::string> as_map(const TxtRecord& txt) {
  return {txt.begin(), txt.end()};
}

TEST(TxtRecord, OneKeyPerField) {
  std::vector<DeviceInfo> devices = {
      {DeviceKind::kMetal, "Apple M5", 17179869184ull, 12713115648ull},
      {DeviceKind::kCpu, "Apple M5", 17179869184ull, 8000000000ull},
  };
  TxtRecord txt = build_txt_record("worker-1", devices);
  auto fields = as_map(txt);

  EXPECT_EQ(fields.size(), txt.size()) << "duplicate TXT keys";
  EXPECT_EQ(fields["txtvers"], "1");
  EXPECT_EQ(fields["proto"], "shardline.v1");
  EXPECT_EQ(fields["id"], "worker-1");
  EXPECT_EQ(fields["devices"], "2");
  EXPECT_EQ(fields["kind"], "metal");
  EXPECT_EQ(fields["device"], "Apple M5");
  EXPECT_EQ(fields["mem_total"], "17179869184");
  EXPECT_EQ(fields["mem_usable"], "12713115648");
}

TEST(TxtRecord, EntriesFitInOneTxtString) {
  std::vector<DeviceInfo> devices = {
      {DeviceKind::kCuda, std::string(400, 'x'), 1, 1},
  };
  for (const auto& [key, value] : build_txt_record(std::string(300, 'w'), devices)) {
    EXPECT_LE(key.size() + 1 + value.size(), 255u) << key;
    EXPECT_EQ(key.find('='), std::string::npos);
  }
}

TEST(TxtRecord, NoDevices) {
  auto fields = as_map(build_txt_record("worker-1", {}));
  EXPECT_EQ(fields["devices"], "0");
  EXPECT_EQ(fields.count("kind"), 0u);
}

}  // namespace
}  // namespace shardline
