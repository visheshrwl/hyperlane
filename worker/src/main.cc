#include <pthread.h>
#include <signal.h>

#include <iostream>
#include <string_view>
#include <thread>

#include "shardline/protocol.h"
#include "shardline/worker_node.h"

namespace {

void print_usage(std::ostream& out) {
  out << "usage: shardline-worker [--listen HOST:PORT] [--advertise] "
         "[--worker-id ID]\n"
         "\n"
         "  --listen HOST:PORT  address to serve on (default 127.0.0.1:50051;\n"
         "                      port 0 picks a free port)\n"
         "  --advertise         announce the worker over mDNS (non-loopback\n"
         "                      listeners only)\n"
         "  --worker-id ID      stable identifier (default HOSTNAME-RANDOM)\n"
         "  --version           print the version and exit\n";
}

std::string bound_address(const std::string& listen_address, int port) {
  auto colon = listen_address.rfind(':');
  std::string host = colon == std::string::npos
                         ? listen_address
                         : listen_address.substr(0, colon);
  return host + ":" + std::to_string(port);
}

}  // namespace

int main(int argc, char* argv[]) {
  shardline::WorkerOptions options;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg = argv[i];
    if (arg == "--listen" && i + 1 < argc) {
      options.listen_address = argv[++i];
    } else if (arg == "--advertise") {
      options.advertise = true;
    } else if (arg == "--worker-id" && i + 1 < argc) {
      options.worker_id = argv[++i];
    } else if (arg == "--version") {
      std::cout << "shardline-worker " << shardline::worker_version() << "\n";
      return 0;
    } else if (arg == "-h" || arg == "--help") {
      print_usage(std::cout);
      return 0;
    } else {
      std::cerr << "unknown or incomplete argument: " << arg << "\n";
      print_usage(std::cerr);
      return 2;
    }
  }

  sigset_t signals;
  sigemptyset(&signals);
  sigaddset(&signals, SIGINT);
  sigaddset(&signals, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &signals, nullptr);

  shardline::WorkerNode node(options);
  if (!node.start()) {
    return 1;
  }
  std::cout << "shardline-worker " << node.worker_id() << " listening on "
            << bound_address(options.listen_address, node.port()) << std::endl;

  std::thread signal_thread([&] {
    int signal_number = 0;
    sigwait(&signals, &signal_number);
    std::cerr << "received signal " << signal_number << ", shutting down\n";
    node.shutdown();
  });

  node.wait();
  signal_thread.join();
  return 0;
}
