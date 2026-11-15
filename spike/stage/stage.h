#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace shardline::spike {

struct ModelConfig {
  uint32_t n_layer = 0;
  uint32_t n_embd = 0;
  uint32_t n_ff = 0;
  uint32_t n_head = 0;
  uint32_t n_head_kv = 0;
  uint32_t head_dim = 0;
  uint32_t n_vocab = 0;
  uint32_t n_ctx_train = 0;
  float rope_base = 10000.0f;
  float rms_eps = 1e-5f;
};

struct StageSpec {
  std::string model_path;
  std::string backend = "cpu";  // cpu | gpu
  int begin = 0;                // first layer, inclusive
  int end = -1;                 // last layer, exclusive; -1 means n_layer
  int n_ctx = 512;
  int threads = 4;
};

// Runs layers [begin, end) of a Llama-family GGUF model on one ggml backend.
// The first stage owns the token embedding; the last stage owns the final
// norm and the output projection.
class Stage {
 public:
  explicit Stage(const StageSpec& spec);
  ~Stage();

  const ModelConfig& config() const;
  const std::string& backend_name() const;
  bool has_embedding() const;
  bool has_head() const;
  size_t weight_bytes() const;

  // The first stage takes token ids; later stages take hidden states laid out
  // as n_tokens rows of n_embd floats. Returns hidden states in the same
  // layout, or n_tokens rows of n_vocab logits from the last stage.
  std::vector<float> forward(const std::vector<int32_t>& tokens,
                             const std::vector<float>& hidden, int n_tokens);

  void reset();
  int n_past() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace shardline::spike
