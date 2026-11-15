#include <arpa/inet.h>
#include <dns_sd.h>
#include <sys/select.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <iostream>
#include <thread>

#include "shardline/discovery.h"

namespace shardline {

namespace {

constexpr std::chrono::milliseconds kPollInterval{200};
constexpr std::chrono::seconds kFirstRetry{1};
constexpr std::chrono::seconds kLastRetry{30};

class DnsSdAdvertiser final : public Advertiser {
 public:
  ~DnsSdAdvertiser() override { stop(); }

  bool start(const std::string& instance_name, uint16_t port,
             const TxtRecord& txt) override {
    stop();
    instance_name_ = instance_name;
    port_ = port;
    txt_ = txt;
    if (!register_service()) {
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
  // Runs on the advertiser thread from DNSServiceProcessResult. A daemon that
  // goes away is reported here as a failed registration, not as an error from
  // the connection, so a failure has to trigger the retry too.
  static void DNSSD_API on_registered(DNSServiceRef, DNSServiceFlags,
                                      DNSServiceErrorType error,
                                      const char* name, const char* type,
                                      const char* domain, void* context) {
    auto* self = static_cast<DnsSdAdvertiser*>(context);
    if (error != kDNSServiceErr_NoError) {
      std::cerr << "mDNS: registration failed with error " << error << "\n";
      self->registration_failed_ = true;
      return;
    }
    self->registered_ = true;
    std::cerr << "mDNS: advertised '" << name << "' as " << type << domain
              << "\n";
  }

  bool register_service() {
    registered_ = false;
    registration_failed_ = false;
    TXTRecordRef record;
    TXTRecordCreate(&record, 0, nullptr);
    for (const auto& [key, value] : txt_) {
      if (TXTRecordSetValue(&record, key.c_str(),
                            static_cast<uint8_t>(value.size()),
                            value.data()) != kDNSServiceErr_NoError) {
        std::cerr << "mDNS: cannot add TXT key '" << key << "'\n";
        TXTRecordDeallocate(&record);
        return false;
      }
    }

    DNSServiceErrorType error = DNSServiceRegister(
        &ref_, 0, kDNSServiceInterfaceIndexAny, instance_name_.c_str(),
        kServiceType, nullptr, nullptr, htons(port_),
        TXTRecordGetLength(&record), TXTRecordGetBytesPtr(&record),
        &DnsSdAdvertiser::on_registered, this);
    TXTRecordDeallocate(&record);
    if (error != kDNSServiceErr_NoError) {
      std::cerr << "mDNS: DNSServiceRegister failed with error " << error
                << " (is the mDNS daemon running?)\n";
      ref_ = nullptr;
      return false;
    }
    return true;
  }

  // Serves the daemon connection for one poll interval. False when the
  // connection is unusable or the registration failed, so the service has to
  // be registered again.
  bool pump() {
    int fd = DNSServiceRefSockFD(ref_);
    fd_set readable;
    FD_ZERO(&readable);
    FD_SET(fd, &readable);
    timeval timeout{0, 200000};
    int ready = select(fd + 1, &readable, nullptr, nullptr, &timeout);
    if (ready < 0) {
      return errno == EINTR;
    }
    if (ready > 0 && DNSServiceProcessResult(ref_) != kDNSServiceErr_NoError) {
      return false;
    }
    return !registration_failed_;
  }

  // Sleeps in short slices so stop() is not kept waiting; false once stopped.
  bool pause(std::chrono::seconds total) {
    for (std::chrono::milliseconds waited{0}; waited < total && running_;
         waited += kPollInterval) {
      std::this_thread::sleep_for(kPollInterval);
    }
    return running_;
  }

  // The daemon can go away while the worker keeps serving (an avahi-daemon
  // restart, for example), so a lost registration is retried with backoff
  // instead of leaving the worker silently undiscoverable. The wait grows
  // only while registrations keep failing; one that was confirmed starts
  // again from the shortest wait.
  void run() {
    auto retry = kFirstRetry;
    while (running_) {
      if (ref_ == nullptr) {
        if (!pause(retry)) {
          return;
        }
        if (!register_service()) {
          retry = std::min(retry * 2, kLastRetry);
          continue;
        }
      }
      if (!pump()) {
        retry = registered_ ? kFirstRetry : std::min(retry * 2, kLastRetry);
        std::cerr << "mDNS: advertisement lost, registering again in "
                  << retry.count() << " s\n";
        DNSServiceRefDeallocate(ref_);
        ref_ = nullptr;
      }
    }
  }

  std::string instance_name_;
  uint16_t port_ = 0;
  TxtRecord txt_;
  DNSServiceRef ref_ = nullptr;
  std::atomic<bool> running_{false};
  std::atomic<bool> registered_{false};
  std::atomic<bool> registration_failed_{false};
  std::thread thread_;
};

}  // namespace

std::unique_ptr<Advertiser> make_mdns_advertiser() {
  return std::make_unique<DnsSdAdvertiser>();
}

}  // namespace shardline
