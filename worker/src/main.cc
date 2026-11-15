#include <pthread.h>
#include <signal.h>

#include <charconv>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <thread>

#include "shardline/protocol.h"
#include "shardline/worker_node.h"

namespace {

void print_usage(std::ostream& out) {
  out << "usage: shardline-worker [--listen HOST:PORT] [--advertise] "
         "[--worker-id ID]\n"
         "                        [--model-dir DIR] [--threads N]\n"
         "\n"
         "  --listen HOST:PORT  address to serve on (default 127.0.0.1:50051;\n"
         "                      port 0 picks a free port)\n"
         "  --advertise         announce the worker over mDNS (non-loopback\n"
         "                      listeners only); exits with an error if the\n"
         "                      mDNS daemon is unavailable at startup\n"
         "  --worker-id ID      stable identifier (default HOSTNAME-RANDOM)\n"
         "  --model-dir DIR     where models named <sha256>.gguf are kept\n"
         "                      (default $SHARDLINE_MODEL_DIR, otherwise\n"
         "                      ~/.cache/shardline/models)\n"
         "  --threads N         CPU threads for model layers (default 4)\n"
         "  --version           print the version and exit\n";
}

std::string default_model_dir() {
  const char* configured = std::getenv("SHARDLINE_MODEL_DIR");
  if (configured != nullptr && *configured != '\0') {
    return configured;
  }
  const char* home = std::getenv("HOME");
  if (home != nullptr && *home != '\0') {
    return std::string(home) + "/.cache/shardline/models";
  }
  return "";
}

bool parse_threads(std::string_view text, int* threads) {
  int value = 0;
  auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc() || result.ptr != text.data() + text.size() ||
      value < 1 || value > 256) {
    return false;
  }
  *threads = value;
  return true;
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
  options.model_dir = default_model_dir();
  for (int i = 1; i < argc; ++i) {
    std::string_view arg = argv[i];
    if (arg == "--listen" && i + 1 < argc) {
      options.listen_address = argv[++i];
    } else if (arg == "--advertise") {
      options.advertise = true;
    } else if (arg == "--worker-id" && i + 1 < argc) {
      options.worker_id = argv[++i];
    } else if (arg == "--model-dir" && i + 1 < argc) {
      options.model_dir = argv[++i];
    } else if (arg == "--threads" && i + 1 < argc) {
      if (!parse_threads(argv[++i], &options.threads)) {
        std::cerr << "--threads takes a number from 1 to 256\n";
        return 2;
      }
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
