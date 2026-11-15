#include "shardline/discovery.h"

#include "shardline/protocol.h"

namespace shardline {

namespace {

constexpr size_t kMaxTxtEntry = 255;

void add(TxtRecord& txt, const std::string& key, std::string value) {
  size_t room = kMaxTxtEntry - key.size() - 1;
  txt.emplace_back(key, truncate_utf8(std::move(value), room));
}

}  // namespace

std::string truncate_utf8(std::string value, size_t max_bytes) {
  if (value.size() <= max_bytes) {
    return value;
  }
  size_t cut = max_bytes;
  while (cut > 0 && (static_cast<unsigned char>(value[cut]) & 0xC0) == 0x80) {
    --cut;
  }
  value.resize(cut);
  return value;
}

std::string instance_name_for(const std::string& worker_id) {
  return truncate_utf8(worker_id, kMaxInstanceName);
}

bool advertise_worker(Advertiser& advertiser, const std::string& worker_id,
                      uint16_t port, const std::vector<DeviceInfo>& devices) {
  return advertiser.start(instance_name_for(worker_id), port,
                          build_txt_record(worker_id, devices));
}

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
