#include "worker.h"
#include <avahi-client/client.h>
#include <avahi-client/publish.h>
#include <avahi-common/alternative.h>
#include <avahi-common/error.h>
#include <avahi-common/timeval.h>
#include <avahi-common/malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <sstream>

namespace hyperlane::worker {

// Forward declarations
static void entry_group_callback(AvahiEntryGroup* g,
                                  AvahiEntryGroupState state, void* userdata);
static void client_callback(AvahiClient* c, AvahiClientState state, void* userdata);

class AvahiContext {
 public:
  AvahiClient* client = nullptr;
  AvahiEntryGroup* group = nullptr;
  AvahiPoll* poll = nullptr;
  ServiceDiscovery* service_discovery = nullptr;
  std::string service_name;
};

static void entry_group_callback(AvahiEntryGroup* g,
                                  AvahiEntryGroupState state, void* userdata) {
  AvahiContext* ctx = static_cast<AvahiContext*>(userdata);
  switch (state) {
    case AVAHI_ENTRY_GROUP_ESTABLISHED:
      std::cout << "[Avahi] Service '" << ctx->service_name << "' successfully established\n";
      break;
    case AVAHI_ENTRY_GROUP_COLLISION: {
      char* n = avahi_alternative_service_name(ctx->service_name.c_str());
      ctx->service_name = n;
      avahi_free(n);
      std::cerr << "[Avahi] Service name collision, renamed to '" << ctx->service_name << "'\n";
      break;
    }
    case AVAHI_ENTRY_GROUP_FAILURE:
      std::cerr << "[Avahi] Entry group failed: "
                << avahi_strerror(avahi_client_errno(ctx->client)) << "\n";
      break;
    case AVAHI_ENTRY_GROUP_UNCOMMITED:
    case AVAHI_ENTRY_GROUP_REGISTERING:
      break;
  }
}

static void client_callback(AvahiClient* c, AvahiClientState state,
                            void* userdata) {
  AvahiContext* ctx = static_cast<AvahiContext*>(userdata);
  switch (state) {
    case AVAHI_CLIENT_S_RUNNING:
      std::cout << "[Avahi] Client running\n";
      break;
    case AVAHI_CLIENT_FAILURE:
      std::cerr << "[Avahi] Client failure: "
                << avahi_strerror(avahi_client_errno(c)) << "\n";
      break;
    case AVAHI_CLIENT_S_COLLISION:
    case AVAHI_CLIENT_S_REGISTERING:
      std::cout << "[Avahi] Client: " << (state == AVAHI_CLIENT_S_COLLISION ? "Collision" : "Registering") << "\n";
      break;
    case AVAHI_CLIENT_CONNECTING:
      std::cout << "[Avahi] Client connecting\n";
      break;
  }
}

ServiceDiscovery::ServiceDiscovery(const std::string& hostname,
                                   uint16_t grpc_port, const GPUStats& stats)
    : hostname_(hostname), grpc_port_(grpc_port), gpu_stats_(stats),
      avahi_context_(nullptr) {
}

ServiceDiscovery::~ServiceDiscovery() {
  unpublish_service();
}

bool ServiceDiscovery::publish_service() {
  int error = 0;

  // Create Avahi context
  avahi_context_ = new AvahiContext();
  avahi_context_->service_discovery = this;

  // Create client
  avahi_context_->client =
      avahi_client_new(avahi_simple_poll_get(avahi_context_->poll), 0,
                      client_callback, avahi_context_, &error);

  if (!avahi_context_->client) {
    std::cerr << "[Avahi] Failed to create client: " << avahi_strerror(error)
              << "\n";
    return false;
  }

  // Create entry group
  avahi_context_->group = avahi_entry_group_new(avahi_context_->client,
                                                  entry_group_callback,
                                                  avahi_context_);
  if (!avahi_context_->group) {
    std::cerr << "[Avahi] Failed to create entry group\n";
    avahi_client_free(avahi_context_->client);
    return false;
  }

  // Build service name
  avahi_context_->service_name = "Hyperlane Worker " + hostname_;

  // Build TXT record with GPU stats
  std::ostringstream txt_stream;
  txt_stream << "gpu_name=" << gpu_stats_.gpu_name
             << " total_memory=" << gpu_stats_.total_memory
             << " free_memory=" << gpu_stats_.free_memory;
  std::string txt_record = txt_stream.str();

  // Add service
  error = avahi_entry_group_add_service(
      avahi_context_->group, AVAHI_IF_UNSPEC, AVAHI_PROTO_UNSPEC, 0,
      avahi_context_->service_name.c_str(), "_hyperlane._tcp", nullptr,
      hostname_.c_str(), grpc_port_, txt_record.c_str(), nullptr);

  if (error < 0) {
    std::cerr << "[Avahi] Failed to add service: " << avahi_strerror(error)
              << "\n";
    avahi_entry_group_free(avahi_context_->group);
    avahi_client_free(avahi_context_->client);
    return false;
  }

  // Commit
  error = avahi_entry_group_commit(avahi_context_->group);
  if (error < 0) {
    std::cerr << "[Avahi] Failed to commit entry group: " << avahi_strerror(error)
              << "\n";
    avahi_entry_group_free(avahi_context_->group);
    avahi_client_free(avahi_context_->client);
    return false;
  }

  std::cout << "[Avahi] Service published: " << avahi_context_->service_name
            << " on port " << grpc_port_ << "\n";
  return true;
}

bool ServiceDiscovery::unpublish_service() {
  if (avahi_context_) {
    if (avahi_context_->group) {
      avahi_entry_group_free(avahi_context_->group);
    }
    if (avahi_context_->client) {
      avahi_client_free(avahi_context_->client);
    }
    delete avahi_context_;
    avahi_context_ = nullptr;
  }
  std::cout << "[Avahi] Service unpublished\n";
  return true;
}

}  // namespace hyperlane::worker
