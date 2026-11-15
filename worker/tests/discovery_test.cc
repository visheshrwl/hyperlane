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

bool is_valid_utf8(const std::string& text) {
  size_t i = 0;
  while (i < text.size()) {
    unsigned char lead = static_cast<unsigned char>(text[i]);
    size_t length = 0;
    if (lead < 0x80) {
      length = 1;
    } else if ((lead >> 5) == 0x6) {
      length = 2;
    } else if ((lead >> 4) == 0xE) {
      length = 3;
    } else if ((lead >> 3) == 0x1E) {
      length = 4;
    }
    if (length == 0 || i + length > text.size()) {
      return false;
    }
    for (size_t k = 1; k < length; ++k) {
      if ((static_cast<unsigned char>(text[i + k]) & 0xC0) != 0x80) {
        return false;
      }
    }
    i += length;
  }
  return true;
}

TEST(TruncateUtf8, LeavesShortValuesAlone) {
  EXPECT_EQ(truncate_utf8("abc", 3), "abc");
  EXPECT_EQ(truncate_utf8("abc", 10), "abc");
}

TEST(TruncateUtf8, CutsAsciiAtTheLimit) {
  EXPECT_EQ(truncate_utf8("abcdef", 4), "abcd");
}

TEST(TruncateUtf8, NeverSplitsASequence) {
  const std::string two_byte = "a\xC3\xA9";                // "aé"
  EXPECT_EQ(truncate_utf8(two_byte, 3), two_byte);
  EXPECT_EQ(truncate_utf8(two_byte, 2), "a");
  const std::string four_byte = "\xF0\x9F\x98\x80";        // one emoji
  EXPECT_EQ(truncate_utf8(four_byte + "x", 3), "");
  EXPECT_EQ(truncate_utf8(four_byte + "x", 4), four_byte);
}

TEST(TxtRecord, MultibyteValuesAreCutOnACodePoint) {
  std::string id;
  while (id.size() < 300) {
    id += "\xC3\xA9";
  }
  std::vector<DeviceInfo> devices = {{DeviceKind::kCuda, id, 1, 1}};
  for (const auto& [key, value] : build_txt_record(id, devices)) {
    EXPECT_LE(key.size() + 1 + value.size(), 255u) << key;
    EXPECT_TRUE(is_valid_utf8(value)) << key;
  }
}

TEST(InstanceName, KeepsShortIdsAsTheyAre) {
  EXPECT_EQ(instance_name_for("mac-mini-1"), "mac-mini-1");
}

TEST(InstanceName, FitsOneDnsLabel) {
  EXPECT_EQ(instance_name_for(std::string(100, 'w')).size(), kMaxInstanceName);
  std::string id;
  while (id.size() < 100) {
    id += "\xC3\xA9";
  }
  std::string name = instance_name_for(id);
  EXPECT_LE(name.size(), kMaxInstanceName);
  EXPECT_TRUE(is_valid_utf8(name));
}

class FakeAdvertiser final : public Advertiser {
 public:
  explicit FakeAdvertiser(bool succeeds) : succeeds_(succeeds) {}
  bool start(const std::string& instance_name, uint16_t port,
             const TxtRecord& txt) override {
    instance_name_ = instance_name;
    port_ = port;
    txt_ = txt;
    return succeeds_;
  }
  void stop() override {}

  bool succeeds_;
  std::string instance_name_;
  uint16_t port_ = 0;
  TxtRecord txt_;
};

TEST(AdvertiseWorker, RegistersALegalNameAndKeepsTheFullIdInTxt) {
  FakeAdvertiser advertiser(true);
  std::string id(100, 'w');
  std::vector<DeviceInfo> devices = {{DeviceKind::kCpu, "CPU", 1, 1}};

  EXPECT_TRUE(advertise_worker(advertiser, id, 50051, devices));

  EXPECT_EQ(advertiser.instance_name_.size(), kMaxInstanceName);
  EXPECT_EQ(advertiser.port_, 50051);
  EXPECT_EQ(as_map(advertiser.txt_)["id"], id);
}

TEST(AdvertiseWorker, ReportsARefusedRegistration) {
  FakeAdvertiser advertiser(false);
  EXPECT_FALSE(advertise_worker(advertiser, "worker-1", 50051, {}));
}

}  // namespace
}  // namespace shardline
