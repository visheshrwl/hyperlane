#include "tensor_receiver.h"
#include "cuda_ops.h"
#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <vector>

namespace hyperlane::worker {

TensorReceiver::TensorReceiver(uint16_t listen_port)
    : listen_port_(listen_port),
      listen_socket_fd_(-1),
      client_socket_fd_(-1),
      pinned_recv_buffer_(nullptr),
      pinned_recv_buffer_size_(0),
      h2d_stream_(nullptr),
      compute_stream_(nullptr) {
  // Allocate pinned memory
  cudaMallocHost(&pinned_recv_buffer_, MAX_PINNED_SIZE);
  pinned_recv_buffer_size_ = MAX_PINNED_SIZE;

  // Create CUDA streams
  cudaStreamCreate(&h2d_stream_);
  cudaStreamCreate(&compute_stream_);
}

TensorReceiver::~TensorReceiver() {
  shutdown_flag_ = true;

  if (recv_thread_ && recv_thread_->joinable()) {
    recv_thread_->join();
  }

  if (client_socket_fd_ >= 0) {
    close(client_socket_fd_);
  }

  if (listen_socket_fd_ >= 0) {
    close(listen_socket_fd_);
  }

  if (pinned_recv_buffer_) {
    cudaFreeHost(pinned_recv_buffer_);
  }

  if (h2d_stream_) {
    cudaStreamDestroy(h2d_stream_);
  }

  if (compute_stream_) {
    cudaStreamDestroy(compute_stream_);
  }
}

void TensorReceiver::start_listening() {
  listen_socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_socket_fd_ < 0) {
    std::cerr << "[TensorReceiver] socket() failed\n";
    return;
  }

  // Allow reuse
  int opt = 1;
  setsockopt(listen_socket_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(listen_port_);

  if (bind(listen_socket_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    std::cerr << "[TensorReceiver] bind() failed\n";
    close(listen_socket_fd_);
    listen_socket_fd_ = -1;
    return;
  }

  if (listen(listen_socket_fd_, 1) < 0) {
    std::cerr << "[TensorReceiver] listen() failed\n";
    close(listen_socket_fd_);
    listen_socket_fd_ = -1;
    return;
  }

  std::cout << "[TensorReceiver] Listening on port " << listen_port_ << "\n";

  // Start background thread
  recv_thread_ = std::make_unique<std::thread>([this] { recv_worker_thread(); });
}

int TensorReceiver::async_recv_tensor(void** device_ptr_out) {
  int handle_id;
  {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    handle_id = next_handle_id_++;
    RecvHandle handle;
    handle.handle_id = handle_id;
    cudaEventCreate(&handle.event);
    active_handles_[handle_id] = handle;
  }

  std::cout << "[TensorReceiver] Async recv handle: " << handle_id << "\n";

  // TODO: Allocate GPU buffer for output
  // TODO: Get device_ptr from active_handles_[handle_id].device_ptr

  if (device_ptr_out) {
    *device_ptr_out = nullptr;  // Placeholder
  }

  return handle_id;
}

bool TensorReceiver::is_recv_complete(int handle) {
  std::lock_guard<std::mutex> lock(handle_mutex_);
  auto it = active_handles_.find(handle);
  if (it == active_handles_.end()) {
    return false;
  }
  return it->second.complete;
}

void TensorReceiver::wait_recv_complete(int handle) {
  while (!is_recv_complete(handle)) {
    usleep(10);
  }
}

void TensorReceiver::recv_worker_thread() {
  while (!shutdown_flag_) {
    // Accept connection
    if (client_socket_fd_ < 0 && listen_socket_fd_ >= 0) {
      struct sockaddr_in client_addr;
      socklen_t client_len = sizeof(client_addr);

      client_socket_fd_ =
          accept(listen_socket_fd_, (struct sockaddr*)&client_addr, &client_len);
      if (client_socket_fd_ < 0) {
        usleep(100);
        continue;
      }

      std::cout << "[TensorReceiver] Accepted connection\n";
    }

    // Receive data
    if (client_socket_fd_ >= 0) {
      uint8_t buffer[65536];
      ssize_t n = recv(client_socket_fd_, buffer, sizeof(buffer), 0);

      if (n > 0) {
        std::vector<uint8_t> data(buffer, buffer + n);
        {
          std::lock_guard<std::mutex> lock(recv_queue_mutex_);
          recv_queue_.push(std::move(data));
        }
        std::cout << "[TensorReceiver] Received " << n << " bytes\n";
      } else if (n == 0) {
        // Connection closed
        close(client_socket_fd_);
        client_socket_fd_ = -1;
      } else {
        usleep(10);
      }
    } else {
      usleep(100);
    }
  }
}

}  // namespace hyperlane::worker
