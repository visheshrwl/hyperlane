#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <memory>
#include <cstdint>

namespace py = pybind11;

namespace hyperlane::pybind {

/**
 * Zero-copy tensor socket bridge for Python.
 * Sends PyTorch/NumPy tensors directly to remote workers without GIL contention.
 */
class PyTensorSocket {
 public:
  PyTensorSocket(const std::string& address, uint16_t port);
  ~PyTensorSocket();

  bool connect();
  void disconnect();

  /**
   * Send a NumPy array or PyTorch tensor asynchronously.
   * The tensor is pinned, quantized (if FP16), and sent in the background.
   * Returns a handle for polling completion.
   */
  int async_send(py::array_t<uint16_t>& tensor, const std::string& name);

  bool is_complete(int handle);
  void wait_complete(int handle);

 private:
  std::string address_;
  uint16_t port_;
  int socket_fd_ = -1;
};

}  // namespace hyperlane::pybind
