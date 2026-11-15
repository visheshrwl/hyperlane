#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "shardline/devices.h"

namespace shardline {

inline constexpr char kServiceType[] = "_shardline._tcp";
inline constexpr char kTxtVersion[] = "1";

using TxtRecord = std::vector<std::pair<std::string, std::string>>;

// One TXT key per field, describing the first (preferred) device. The values
// are hints for discovery; GetStats is authoritative.
TxtRecord build_txt_record(const std::string& worker_id,
                           const std::vector<DeviceInfo>& devices);

class Advertiser {
 public:
  virtual ~Advertiser() = default;
  virtual bool start(const std::string& instance_name, uint16_t port,
                     const TxtRecord& txt) = 0;
  virtual void stop() = 0;
};

// Returns nullptr when the worker was built without mDNS support.
std::unique_ptr<Advertiser> make_mdns_advertiser();

}  // namespace shardline
