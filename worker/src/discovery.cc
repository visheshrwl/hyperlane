#include "shardline/discovery.h"

#include "shardline/protocol.h"

namespace shardline {

namespace {

constexpr size_t kMaxTxtEntry = 255;

void add(TxtRecord& txt, const std::string& key, std::string value) {
  size_t room = kMaxTxtEntry - key.size() - 1;
  if (value.size() > room) {
    value.resize(room);
  }
  txt.emplace_back(key, std::move(value));
}

}  // namespace

TxtRecord build_txt_record(const std::string& worker_id,
                           const std::vector<DeviceInfo>& devices) {
  TxtRecord txt;
  add(txt, "txtvers", kTxtVersion);
  add(txt, "proto", kProtocol);
  add(txt, "id", worker_id);
  add(txt, "devices", std::to_string(devices.size()));
  if (!devices.empty()) {
    const DeviceInfo& device = devices.front();
    add(txt, "kind", device_kind_name(device.kind));
    add(txt, "device", device.name);
    add(txt, "mem_total", std::to_string(device.total_memory_bytes));
    add(txt, "mem_usable", std::to_string(device.usable_memory_bytes));
  }
  return txt;
}

#if !defined(SHARDLINE_HAVE_MDNS)
std::unique_ptr<Advertiser> make_mdns_advertiser() { return nullptr; }
#endif

}  // namespace shardline
