#pragma once

#include <memory>
#include <string>
#include <map>
#include <vector>
#include <onnxruntime_cxx_api.h>

namespace hyperlane::worker {

/**
 * ONNX Runtime session wrapper for executing model shards.
 */
class ONNXSessionWrapper {
 public:
  explicit ONNXSessionWrapper(const std::string& model_path);
  ~ONNXSessionWrapper();

  bool initialize();
  
  // Run inference with GPU tensors
  bool run(const std::vector<const void*>& input_buffers,
           const std::vector<int64_t>& input_size,
           std::vector<void*>& output_buffers);

  const std::vector<std::string>& get_input_names() const {
    return input_names_;
  }
  const std::vector<std::string>& get_output_names() const {
    return output_names_;
  }

 private:
  std::string model_path_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::Env> env_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};

}  // namespace hyperlane::worker
