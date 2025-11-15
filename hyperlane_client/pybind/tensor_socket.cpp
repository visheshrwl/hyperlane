#include "tensor_socket.h"
#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <queue>

namespace hyperlane::pybind {

// Global state for async handles (in production, use a thread-safe map)
static std::mutex g_handle_mutex;
static std::map<int, bool> g_handle_complete;
static int g_next_handle_id = 0;

PyTensorSocket::PyTensorSocket(const std::string& address, uint16_t port)
    : address_(address), port_(port) {
}

PyTensorSocket::~PyTensorSocket() {
  disconnect();
}

bool PyTensorSocket::connect() {
  if (socket_fd_ >= 0) {
    return true;
  }

  socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd_ < 0) {
    std::cerr << "[PyTensorSocket] socket() failed\n";
    return false;
  }

  struct sockaddr_in server_addr;
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port_);

  if (inet_pton(AF_INET, address_.c_str(), &server_addr.sin_addr) <= 0) {
    std::cerr << "[PyTensorSocket] inet_pton() failed\n";
    close(socket_fd_);
    socket_fd_ = -1;
    return false;
  }

  if (::connect(socket_fd_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    std::cerr << "[PyTensorSocket] connect() failed\n";
    close(socket_fd_);
    socket_fd_ = -1;
    return false;
  }

  std::cout << "[PyTensorSocket] Connected to " << address_ << ":" << port_
            << "\n";
  return true;
}

void PyTensorSocket::disconnect() {
  if (socket_fd_ >= 0) {
    close(socket_fd_);
    socket_fd_ = -1;
  }
}

int PyTensorSocket::async_send(py::array_t<uint16_t>& tensor,
                               const std::string& name) {
  if (socket_fd_ < 0) {
    std::cerr << "[PyTensorSocket] Not connected\n";
    return -1;
  }

  // Extract buffer info (zero-copy access to NumPy array)
  auto buf = tensor.request();
  const uint16_t* data = static_cast<const uint16_t*>(buf.ptr);
  size_t size_elements = tensor.size();

  std::cout << "[PyTensorSocket] Async send: " << name << " (" << size_elements
            << " elements)\n";

  // Allocate handle
  int handle_id;
  {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    handle_id = g_next_handle_id++;
    g_handle_complete[handle_id] = false;
  }

  // In a real implementation, we'd:
  // 1. Quantize the FP16 tensor to INT4 (via CUDA if on GPU)
  // 2. Send in background thread
  // 3. Mark complete when done
  //
  // For now, send synchronously and mark complete
  if (socket_fd_ >= 0) {
    size_t bytes = size_elements * sizeof(uint16_t);
    ssize_t sent = send(socket_fd_, data, bytes, 0);
    if (sent < 0) {
      std::cerr << "[PyTensorSocket] send() failed\n";
      return -1;
    }
    std::cout << "[PyTensorSocket] Sent " << sent << " bytes\n";
  }

  {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    g_handle_complete[handle_id] = true;
  }

  return handle_id;
}

bool PyTensorSocket::is_complete(int handle) {
  std::lock_guard<std::mutex> lock(g_handle_mutex);
  auto it = g_handle_complete.find(handle);
  return it != g_handle_complete.end() && it->second;
}

void PyTensorSocket::wait_complete(int handle) {
  while (!is_complete(handle)) {
    usleep(10);
  }
}

}  // namespace hyperlane::pybind

// pybind11 module binding
PYBIND11_MODULE(hyperlane_tensor_socket, m) {
  m.doc() = "Hyperlane zero-copy tensor socket extension";

  py::class_<hyperlane::pybind::PyTensorSocket>(m, "TensorSocket")
      .def(py::init<const std::string&, uint16_t>())
      .def("connect", &hyperlane::pybind::PyTensorSocket::connect)
      .def("disconnect", &hyperlane::pybind::PyTensorSocket::disconnect)
      .def("async_send", &hyperlane::pybind::PyTensorSocket::async_send)
      .def("is_complete", &hyperlane::pybind::PyTensorSocket::is_complete)
      .def("wait_complete", &hyperlane::pybind::PyTensorSocket::wait_complete);
}
