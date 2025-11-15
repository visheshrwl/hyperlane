#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <thread>
#include <queue>
#include <mutex>

namespace hyperlane::worker {

/**
 * Async handle for tracking tensor transmission state.
 */
struct SendHandle {
  int handle_id;
  bool complete = false;
  cudaEvent_t event = nullptr;
};

/**
 * Asynchronous tensor sender for inter-worker communication.
 *
 * Flow:
 *  1. Quantize FP16->INT4 in VRAM (custom CUDA kernel)
 *  2. Async D2H copy to pinned host buffer (dedicated stream)
 *  3. Non-blocking socket send from pinned buffer (DMA-friendly)
 *  4. CUDA events signal completion
 */
class TensorSender {
 public:
  TensorSender(const std::string& next_worker_address, uint16_t next_port);
  ~TensorSender();

  // Connect to next worker
  bool connect();

  /**
   * Async send: quantize, D2H, transmit.
   * Returns a handle for polling/waiting.
   */
  int async_send_tensor(const void* device_ptr, const std::vector<int64_t>& shape,
                        const std::string& tensor_name);

  bool is_send_complete(int handle);
  void wait_send_complete(int handle);

  void set_next_worker(const std::string& address, uint16_t port);

 private:
  std::string next_worker_address_;
  uint16_t next_port_;
  int socket_fd_;

  // Pinned memory for D2H staging
  void* pinned_buffer_;
  size_t pinned_buffer_size_;
  static const size_t MAX_PINNED_SIZE = 512 * 1024 * 1024;  // 512 MB

  // CUDA streams for async operations
  cudaStream_t d2h_stream_;
  cudaStream_t compute_stream_;

  // Handle tracking
  std::mutex handle_mutex_;
  std::map<int, SendHandle> active_handles_;
  int next_handle_id_ = 0;

  // Background thread for socket I/O
  std::unique_ptr<std::thread> send_thread_;
  std::queue<std::pair<int, std::vector<uint8_t>>> send_queue_;
  std::mutex send_queue_mutex_;
  bool shutdown_flag_ = false;

  void send_worker_thread();
};

}  // namespace hyperlane::worker
