#include "worker.h"
#include <iostream>

int main(int argc, char* argv[]) {
  uint16_t grpc_port = 50051;

  if (argc > 1) {
    grpc_port = static_cast<uint16_t>(std::stoi(argv[1]));
  }

  std::cout << "Starting Hyperlane Worker on port " << grpc_port << "\n";

  hyperlane::worker::Worker worker(grpc_port);

  if (!worker.initialize()) {
    std::cerr << "Failed to initialize worker\n";
    return 1;
  }

  worker.run();
  worker.shutdown();

  return 0;
}
