#pragma once

#include <string>
#include <cstdint>
#include <memory>
#include <thread>
#include <queue>
#include <mutex>
#include <cuda_runtime.h>

namespace hyperlane::worker {

/**
 * Async handle for tracking tensor reception state.
 */
struct RecvHandle {
  int handle_id;
  bool complete = false;
  void* device_ptr = nullptr;
  cudaEvent_t event = nullptr;
};

/**
 * Asynchronous tensor receiver for pipeline input.
 *
 * Flow:
 *  1. Accept connection from previous worker
 *  2. Non-blocking socket recv into pinned host buffer
 *  3. Async H2D copy from pinned buffer to GPU VRAM
 *  4. Dequantize INT4->FP16 in VRAM
 *  5. CUDA events signal completion
 */
class TensorReceiver {
 public:
  explicit TensorReceiver(uint16_t listen_port);
  ~TensorReceiver();

  // Start listening for incoming tensor connections
  void start_listening();

  /**
   * Async receive: poll for socket data, H2D, dequantize.
   * Returns a handle for polling/waiting and gets device_ptr_out.
   */
  int async_recv_tensor(void** device_ptr_out);

  bool is_recv_complete(int handle);
  void wait_recv_complete(int handle);

 private:
  uint16_t listen_port_;
  int listen_socket_fd_;
  int client_socket_fd_;

  // Pinned memory for H2D staging
  void* pinned_recv_buffer_;
  size_t pinned_recv_buffer_size_;
  static const size_t MAX_PINNED_SIZE = 512 * 1024 * 1024;  // 512 MB

  // CUDA streams for async operations
  cudaStream_t h2d_stream_;
  cudaStream_t compute_stream_;

  // Handle tracking
  std::mutex handle_mutex_;
  std::map<int, RecvHandle> active_handles_;
  int next_handle_id_ = 0;

  // Background thread for socket I/O
  std::unique_ptr<std::thread> recv_thread_;
  std::queue<std::vector<uint8_t>> recv_queue_;
  std::mutex recv_queue_mutex_;
  bool shutdown_flag_ = false;

  void recv_worker_thread();
};

}  // namespace hyperlane::worker
