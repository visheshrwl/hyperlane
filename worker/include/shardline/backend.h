#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "shardline/devices.h"

namespace shardline {

enum class ErrorKind {
  kInvalidArgument,
  kNotFound,
  kAlreadyExists,
  kResourceExhausted,
  kFailedPrecondition,
  kUnimplemented,
  kInternal,
};

class BackendError : public std::runtime_error {
 public:
  BackendError(ErrorKind kind, const std::string& message)
      : std::runtime_error(message), kind_(kind) {}
  ErrorKind kind() const { return kind_; }

 private:
  ErrorKind kind_;
};

struct ModelInfo {
  uint32_t n_layer = 0;
  uint32_t n_embd = 0;
  uint32_t n_vocab = 0;
  uint32_t n_ctx_train = 0;
};

struct ShardSpec {
  std::string model_path;
  DeviceKind device = DeviceKind::kCpu;
  uint32_t layer_begin = 0;
  uint32_t layer_end = 0;  // exclusive; 0 means through the last layer
  // Tokens of KV cache per session. 0 means 512, or the model's trained
  // context when that is smaller.
  uint32_t context_length = 0;
  uint32_t max_sessions = 8;
  int threads = 4;
};

struct ShardInfo {
  uint32_t layer_begin = 0;
  uint32_t layer_end = 0;
  bool has_embedding = false;
  bool has_head = false;
  uint64_t weight_bytes = 0;
  uint32_t context_length = 0;  // as loaded, once the default is resolved
  std::string backend;
};

struct StageInput {
  std::vector<int32_t> tokens;  // the shard that holds the embedding
  std::vector<float> hidden;    // every other shard: n_tokens rows of n_embd
  bool last_row_only = true;    // the shard that holds the head: skip the other rows
};

struct StageOutput {
  std::vector<float> data;  // row-major, rows x (n_embd or n_vocab)
  uint32_t rows = 0;
  bool is_logits = false;
};

// Runs layers [begin, end) of a Llama-family model for any number of sessions,
// each with its own KV cache. A backend is not thread safe; callers serialise
// access.
class Backend {
 public:
  virtual ~Backend() = default;

  virtual const ModelInfo& model() const = 0;
  virtual const ShardInfo& shard() const = 0;

  // Throws BackendError.
  virtual void open_session(const std::string& id) = 0;
  virtual void close_session(const std::string& id) = 0;
  virtual bool has_session(const std::string& id) const = 0;

  // `position` is the number of tokens the session already holds. It must
  // match, so a lost or repeated step is an error instead of silent garbage.
  // Throws BackendError.
  virtual StageOutput forward(const std::string& session, uint32_t position,
                              StageInput input) = 0;
};

using BackendFactory =
    std::function<std::unique_ptr<Backend>(const ShardSpec&)>;

// Throws BackendError(kUnimplemented) when the worker was built without ggml.
std::unique_ptr<Backend> load_ggml_backend(const ShardSpec& spec);

// <model_dir>/<sha256>.gguf. Throws BackendError(kInvalidArgument) unless
// sha256 is 64 lowercase hex digits, so a request cannot name another path.
std::string model_path_for_digest(const std::string& model_dir,
                                  const std::string& sha256);

}  // namespace shardline
