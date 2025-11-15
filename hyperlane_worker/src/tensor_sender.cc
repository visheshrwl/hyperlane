#include "tensor_sender.h"
#include "cuda_ops.h"
#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

namespace hyperlane::worker {

TensorSender::TensorSender(const std::string& next_worker_address,
                           uint16_t next_port)
    : next_worker_address_(next_worker_address),
      next_port_(next_port),
      socket_fd_(-1),
      pinned_buffer_(nullptr),
      pinned_buffer_size_(0),
      d2h_stream_(nullptr),
      compute_stream_(nullptr) {
  // Allocate pinned memory
  cudaMallocHost(&pinned_buffer_, MAX_PINNED_SIZE);
  pinned_buffer_size_ = MAX_PINNED_SIZE;

  // Create CUDA streams
  cudaStreamCreate(&d2h_stream_);
  cudaStreamCreate(&compute_stream_);

  // Start background send thread
  send_thread_ = std::make_unique<std::thread>([this] { send_worker_thread(); });
}

TensorSender::~TensorSender() {
  shutdown_flag_ = true;
  if (send_thread_ && send_thread_->joinable()) {
    send_thread_->join();
  }

  if (socket_fd_ >= 0) {
    close(socket_fd_);
  }

  if (pinned_buffer_) {
    cudaFreeHost(pinned_buffer_);
  }

  if (d2h_stream_) {
    cudaStreamDestroy(d2h_stream_);
  }

  if (compute_stream_) {
    cudaStreamDestroy(compute_stream_);
  }
}

bool TensorSender::connect() {
  if (socket_fd_ >= 0) {
    return true;  // Already connected
  }

  socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd_ < 0) {
    std::cerr << "[TensorSender] socket() failed\n";
    return false;
  }

  struct sockaddr_in server_addr;
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(next_port_);

  if (inet_pton(AF_INET, next_worker_address_.c_str(), &server_addr.sin_addr) <= 0) {
    std::cerr << "[TensorSender] inet_pton() failed\n";
    close(socket_fd_);
    socket_fd_ = -1;
    return false;
  }

  if (::connect(socket_fd_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    std::cerr << "[TensorSender] connect() failed\n";
    close(socket_fd_);
    socket_fd_ = -1;
    return false;
  }

  std::cout << "[TensorSender] Connected to " << next_worker_address_ << ":"
            << next_port_ << "\n";
  return true;
}

int TensorSender::async_send_tensor(const void* device_ptr,
                                     const std::vector<int64_t>& shape,
                                     const std::string& tensor_name) {
  int handle_id;
  {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    handle_id = next_handle_id_++;
    SendHandle handle;
    handle.handle_id = handle_id;
    cudaEventCreate(&handle.event);
    active_handles_[handle_id] = handle;
  }

  // Calculate tensor size
  int64_t num_elements = 1;
  for (int64_t dim : shape) {
    num_elements *= dim;
  }

  std::cout << "[TensorSender] Async send: " << tensor_name << " (" << num_elements
            << " elements)\n";

  // TODO: Queue quantization kernel on compute_stream_
  // TODO: Queue D2H copy on d2h_stream_ to pinned_buffer_
  // TODO: Record event
  // TODO: Enqueue to send_queue_ for background thread

  // Placeholder: just mark complete
  {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    active_handles_[handle_id].complete = true;
  }

  return handle_id;
}

bool TensorSender::is_send_complete(int handle) {
  std::lock_guard<std::mutex> lock(handle_mutex_);
  auto it = active_handles_.find(handle);
  if (it == active_handles_.end()) {
    return false;
  }
  return it->second.complete;
}

void TensorSender::wait_send_complete(int handle) {
  while (!is_send_complete(handle)) {
    // Spin or sleep
    usleep(10);
  }
}

void TensorSender::set_next_worker(const std::string& address, uint16_t port) {
  next_worker_address_ = address;
  next_port_ = port;
  if (socket_fd_ >= 0) {
    close(socket_fd_);
    socket_fd_ = -1;
  }
  connect();
}

void TensorSender::send_worker_thread() {
  while (!shutdown_flag_) {
    std::pair<int, std::vector<uint8_t>> item;
    {
      std::lock_guard<std::mutex> lock(send_queue_mutex_);
      if (!send_queue_.empty()) {
        item = std::move(send_queue_.front());
        send_queue_.pop();
      } else {
        // Queue empty, sleep briefly
        usleep(100);
        continue;
      }
    }

    int handle_id = item.first;
    const auto& data = item.second;

    // Send data over socket
    if (socket_fd_ >= 0) {
      ssize_t sent = send(socket_fd_, data.data(), data.size(), 0);
      if (sent < 0) {
        std::cerr << "[TensorSender] send() failed\n";
      } else {
        std::cout << "[TensorSender] Sent " << sent << " bytes\n";
      }
    }

    // Mark complete
    {
      std::lock_guard<std::mutex> lock(handle_mutex_);
      auto it = active_handles_.find(handle_id);
      if (it != active_handles_.end()) {
        it->second.complete = true;
      }
    }
  }
}

}  // namespace hyperlane::worker
