#include <arpa/inet.h>
#include <dns_sd.h>
#include <sys/select.h>

#include <atomic>
#include <cerrno>
#include <iostream>
#include <thread>

#include "shardline/discovery.h"

namespace shardline {

namespace {

class DnsSdAdvertiser final : public Advertiser {
 public:
  ~DnsSdAdvertiser() override { stop(); }

  bool start(const std::string& instance_name, uint16_t port,
             const TxtRecord& txt) override {
    TXTRecordRef record;
    TXTRecordCreate(&record, 0, nullptr);
    for (const auto& [key, value] : txt) {
      if (TXTRecordSetValue(&record, key.c_str(),
                            static_cast<uint8_t>(value.size()),
                            value.data()) != kDNSServiceErr_NoError) {
        std::cerr << "mDNS: cannot add TXT key '" << key << "'\n";
        TXTRecordDeallocate(&record);
        return false;
      }
    }

    DNSServiceErrorType error = DNSServiceRegister(
        &ref_, 0, kDNSServiceInterfaceIndexAny, instance_name.c_str(),
        kServiceType, nullptr, nullptr, htons(port),
        TXTRecordGetLength(&record), TXTRecordGetBytesPtr(&record),
        &DnsSdAdvertiser::on_registered, nullptr);
    TXTRecordDeallocate(&record);
    if (error != kDNSServiceErr_NoError) {
      std::cerr << "mDNS: DNSServiceRegister failed with error " << error
                << " (is the mDNS daemon running?)\n";
      ref_ = nullptr;
      return false;
    }

    running_ = true;
    thread_ = std::thread([this] { run(); });
    return true;
  }

  void stop() override {
    running_ = false;
    if (thread_.joinable()) {
      thread_.join();
    }
    if (ref_ != nullptr) {
      DNSServiceRefDeallocate(ref_);
      ref_ = nullptr;
    }
  }

 private:
  static void DNSSD_API on_registered(DNSServiceRef, DNSServiceFlags,
                                      DNSServiceErrorType error,
                                      const char* name, const char* type,
                                      const char* domain, void*) {
    if (error != kDNSServiceErr_NoError) {
      std::cerr << "mDNS: registration failed with error " << error << "\n";
      return;
    }
    std::cerr << "mDNS: advertised '" << name << "' as " << type << domain
              << "\n";
  }

  void run() {
    int fd = DNSServiceRefSockFD(ref_);
    while (running_) {
      fd_set readable;
      FD_ZERO(&readable);
      FD_SET(fd, &readable);
      timeval timeout{0, 200000};
      int ready = select(fd + 1, &readable, nullptr, nullptr, &timeout);
      if (ready < 0 && errno != EINTR) {
        std::cerr << "mDNS: select failed, advertisement stopped\n";
        return;
      }
      if (ready > 0 &&
          DNSServiceProcessResult(ref_) != kDNSServiceErr_NoError) {
        std::cerr << "mDNS: lost connection to the mDNS daemon\n";
        return;
      }
    }
  }

  DNSServiceRef ref_ = nullptr;
  std::atomic<bool> running_{false};
  std::thread thread_;
};

}  // namespace

std::unique_ptr<Advertiser> make_mdns_advertiser() {
  return std::make_unique<DnsSdAdvertiser>();
}

}  // namespace shardline
