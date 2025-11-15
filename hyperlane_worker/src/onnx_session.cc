#include "onnx_session.h"
#include <iostream>
#include <algorithm>

namespace hyperlane::worker {

ONNXSessionWrapper::ONNXSessionWrapper(const std::string& model_path)
    : model_path_(model_path) {
}

ONNXSessionWrapper::~ONNXSessionWrapper() = default;

bool ONNXSessionWrapper::initialize() {
  try {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "hyperlane");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Enable CUDA execution provider
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    session_options.AppendExecutionProvider_CUDA(cuda_options);
    session_options.AppendExecutionProvider_CPU();

    session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(),
                                               session_options);

    // Introspect model inputs/outputs
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session_->GetInputCount();
    size_t num_outputs = session_->GetOutputCount();

    for (size_t i = 0; i < num_inputs; ++i) {
      auto name = session_->GetInputName(i, allocator);
      input_names_.emplace_back(name);
      allocator.Free(name);
    }

    for (size_t i = 0; i < num_outputs; ++i) {
      auto name = session_->GetOutputName(i, allocator);
      output_names_.emplace_back(name);
      allocator.Free(name);
    }

    std::cout << "[ONNX] Loaded model: " << model_path_
              << " (inputs: " << input_names_.size()
              << ", outputs: " << output_names_.size() << ")\n";

    return true;
  } catch (const Ort::Exception& e) {
    std::cerr << "[ONNX] Failed to load model: " << e.what() << "\n";
    return false;
  }
}

bool ONNXSessionWrapper::run(const std::vector<const void*>& input_buffers,
                             const std::vector<int64_t>& input_size,
                             std::vector<void*>& output_buffers) {
  try {
    // TODO: Build input/output tensors from GPU memory pointers
    // Use Ort::MemoryInfo with CUDA device

    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char*> input_names, output_names;

    for (const auto& name : input_names_) {
      input_names.push_back(name.c_str());
    }
    for (const auto& name : output_names_) {
      output_names.push_back(name.c_str());
    }

    // Placeholder: actual tensor binding requires GPU memory layout
    // session_->Run(Ort::RunOptions{nullptr},
    //               input_names.data(), input_tensors.data(), input_tensors.size(),
    //               output_names.data(), output_tensors.data(), output_tensors.size());

    std::cout << "[ONNX] Run complete\n";
    return true;
  } catch (const Ort::Exception& e) {
    std::cerr << "[ONNX] Run failed: " << e.what() << "\n";
    return false;
  }
}

}  // namespace hyperlane::worker
