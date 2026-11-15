#pragma once

#include <cstdint>
#include <string>

namespace shardline::testing {

struct TinyModelOptions {
  uint32_t n_layer = 4;
  uint32_t n_embd = 32;
  uint32_t n_head = 4;
  uint32_t n_head_kv = 2;
  uint32_t n_ff = 64;
  uint32_t n_vocab = 48;
  uint32_t n_ctx = 64;
  uint32_t seed = 1;
  bool wrong_ffn_down_shape = false;
};

// Writes a randomly initialised Llama GGUF as <dir>/<name> and returns the path.
std::string write_tiny_model(const std::string& dir, const std::string& name,
                             const TinyModelOptions& options = {});

// A new empty directory under the system temp directory.
std::string make_temp_dir();

}  // namespace shardline::testing
