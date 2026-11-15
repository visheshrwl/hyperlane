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

// A DNS-SD instance name is one DNS label, which holds at most 63 bytes.
inline constexpr size_t kMaxInstanceName = 63;

using TxtRecord = std::vector<std::pair<std::string, std::string>>;

// Cuts value to at most max_bytes without splitting a UTF-8 sequence.
std::string truncate_utf8(std::string value, size_t max_bytes);

// The worker id cut to a legal instance name. The full id stays in the TXT
// record under "id".
std::string instance_name_for(const std::string& worker_id);

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

// Starts advertising the worker; false when the advertiser could not register.
bool advertise_worker(Advertiser& advertiser, const std::string& worker_id,
                      uint16_t port, const std::vector<DeviceInfo>& devices);

// Returns nullptr when the worker was built without mDNS support.
std::unique_ptr<Advertiser> make_mdns_advertiser();

}  // namespace shardline
