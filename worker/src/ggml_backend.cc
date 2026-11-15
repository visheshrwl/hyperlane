#include <algorithm>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "shardline/backend.h"

namespace shardline {

namespace {

constexpr uint32_t kDefaultContextLength = 512;

[[noreturn]] void fail(ErrorKind kind, const std::string& message) {
  throw BackendError(kind, message);
}

uint32_t get_u32(const gguf_context* gguf, const std::string& key) {
  int64_t id = gguf_find_key(gguf, key.c_str());
  if (id < 0) {
    fail(ErrorKind::kInvalidArgument, "the model has no GGUF key " + key);
  }
  if (gguf_get_kv_type(gguf, id) != GGUF_TYPE_UINT32) {
    fail(ErrorKind::kInvalidArgument, "GGUF key " + key + " is not a uint32");
  }
  return gguf_get_val_u32(gguf, id);
}

float get_f32(const gguf_context* gguf, const std::string& key) {
  int64_t id = gguf_find_key(gguf, key.c_str());
  if (id < 0) {
    fail(ErrorKind::kInvalidArgument, "the model has no GGUF key " + key);
  }
  if (gguf_get_kv_type(gguf, id) != GGUF_TYPE_FLOAT32) {
    fail(ErrorKind::kInvalidArgument, "GGUF key " + key + " is not a float32");
  }
  return gguf_get_val_f32(gguf, id);
}

// ggml aborts the whole process when an operator gets mismatched shapes, so a
// damaged or foreign model file has to be refused before any graph is built.
void expect_shape(const ggml_tensor* tensor, std::initializer_list<int64_t> ne) {
  int64_t expected[GGML_MAX_DIMS] = {1, 1, 1, 1};
  size_t i = 0;
  for (int64_t n : ne) {
    expected[i++] = n;
  }
  for (int d = 0; d < GGML_MAX_DIMS; ++d) {
    if (tensor->ne[d] != expected[d]) {
      fail(ErrorKind::kInvalidArgument,
           std::string("tensor ") + tensor->name + " has an unexpected shape");
    }
  }
}

void expect_f32(const ggml_tensor* tensor) {
  if (tensor->type != GGML_TYPE_F32) {
    fail(ErrorKind::kInvalidArgument,
         std::string("tensor ") + tensor->name + " must be float32");
  }
}

struct Layer {
  ggml_tensor* attn_norm = nullptr;
  ggml_tensor* wq = nullptr;
  ggml_tensor* wk = nullptr;
  ggml_tensor* wv = nullptr;
  ggml_tensor* wo = nullptr;
  ggml_tensor* ffn_norm = nullptr;
  ggml_tensor* w_gate = nullptr;
  ggml_tensor* w_up = nullptr;
  ggml_tensor* w_down = nullptr;
};

struct ContextGuard {
  explicit ContextGuard(ggml_context* context) : context(context) {}
  ~ContextGuard() {
    if (context != nullptr) {
      ggml_free(context);
    }
  }
  ContextGuard(const ContextGuard&) = delete;
  ContextGuard& operator=(const ContextGuard&) = delete;
  ggml_context* context;
};

struct Session {
  Session() = default;
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  ~Session() {
    if (buffer != nullptr) {
      ggml_backend_buffer_free(buffer);
    }
    if (context != nullptr) {
      ggml_free(context);
    }
  }

  ggml_context* context = nullptr;
  ggml_backend_buffer_t buffer = nullptr;
  std::vector<ggml_tensor*> k_cache;
  std::vector<ggml_tensor*> v_cache;
  uint32_t n_past = 0;
};

// The name ggml registers an accelerator backend under.
const char* accelerator_name(DeviceKind kind) {
  return kind == DeviceKind::kMetal ? "MTL" : "CUDA";
}

class GgmlBackend final : public Backend {
 public:
  explicit GgmlBackend(ShardSpec spec) : spec_(std::move(spec)) {}

  ~GgmlBackend() override {
    sessions_.clear();
    if (galloc_ != nullptr) ggml_gallocr_free(galloc_);
    if (buf_w_ != nullptr) ggml_backend_buffer_free(buf_w_);
    if (ctx_w_ != nullptr) ggml_free(ctx_w_);
    if (backend_ != nullptr) ggml_backend_free(backend_);
    if (meta_ != nullptr) ggml_free(meta_);
    if (gguf_ != nullptr) gguf_free(gguf_);
  }

  // Loading can fail half way, and a destructor only runs for a fully
  // constructed object, so the work is a separate step.
  void load() {
    open_backend();
    read_config();
    load_weights();
    galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (galloc_ == nullptr) {
      fail(ErrorKind::kInternal, "cannot create the graph allocator");
    }
  }

  const ModelInfo& model() const override { return model_; }
  const ShardInfo& shard() const override { return shard_; }

  void open_session(const std::string& id) override {
    if (sessions_.count(id) != 0) {
      fail(ErrorKind::kAlreadyExists, "session " + id + " already exists");
    }
    if (sessions_.size() >= spec_.max_sessions) {
      fail(ErrorKind::kResourceExhausted,
           "this worker holds " + std::to_string(spec_.max_sessions) +
               " sessions already");
    }
    sessions_[id] = make_session();
  }

  void close_session(const std::string& id) override {
    if (sessions_.erase(id) == 0) {
      fail(ErrorKind::kNotFound, "no session " + id);
    }
  }

  bool has_session(const std::string& id) const override {
    return sessions_.count(id) != 0;
  }

  StageOutput forward(const std::string& id, uint32_t position,
                      StageInput input) override {
    auto found = sessions_.find(id);
    if (found == sessions_.end()) {
      fail(ErrorKind::kNotFound, "no session " + id);
    }
    Session& session = *found->second;
    if (position != session.n_past) {
      fail(ErrorKind::kFailedPrecondition,
           "session " + id + " holds " + std::to_string(session.n_past) +
               " tokens, the request starts at " + std::to_string(position));
    }

    const uint32_t n_tokens = count_tokens(input);
    if (static_cast<uint64_t>(session.n_past) + n_tokens > context_length_) {
      fail(ErrorKind::kResourceExhausted,
           "the context holds " + std::to_string(context_length_) +
               " tokens, which this step would exceed");
    }

    const bool last_row_only =
        input.last_row_only && output_ != nullptr && n_tokens > 1;
    const int n_kv = static_cast<int>(session.n_past + n_tokens);
    const int hd = static_cast<int>(head_dim_);
    const int n_embd = static_cast<int>(model_.n_embd);

    constexpr size_t kMaxNodes = 8192;
    ggml_init_params params{
        ggml_tensor_overhead() * kMaxNodes +
            ggml_graph_overhead_custom(kMaxNodes, false),
        nullptr, true};
    ContextGuard guard(ggml_init(params));
    ggml_context* ctx = guard.context;
    if (ctx == nullptr) {
      fail(ErrorKind::kInternal, "cannot create a graph context");
    }
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, kMaxNodes, false);

    ggml_tensor* inp = nullptr;
    ggml_tensor* x = nullptr;
    if (tok_embd_ != nullptr) {
      inp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
      ggml_set_input(inp);
      x = ggml_get_rows(ctx, tok_embd_, inp);
    } else {
      inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
      ggml_set_input(inp);
      x = inp;
    }
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(pos);
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_tokens);
    ggml_set_input(mask);

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    for (size_t i = 0; i < layers_.size(); ++i) {
      const Layer& L = layers_[i];
      ggml_tensor* cur =
          ggml_mul(ctx, ggml_rms_norm(ctx, x, rms_eps_), L.attn_norm);

      ggml_tensor* q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wq, cur), hd,
                                       n_head_, n_tokens);
      ggml_tensor* k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wk, cur), hd,
                                       n_head_kv_, n_tokens);
      ggml_tensor* v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wv, cur), hd,
                                       n_head_kv_, n_tokens);
      q = ggml_rope_ext(ctx, q, pos, nullptr, hd, GGML_ROPE_TYPE_NORMAL,
                        model_.n_ctx_train, rope_base_, 1.0f, 0.0f, 1.0f, 0.0f,
                        0.0f);
      k = ggml_rope_ext(ctx, k, pos, nullptr, hd, GGML_ROPE_TYPE_NORMAL,
                        model_.n_ctx_train, rope_base_, 1.0f, 0.0f, 1.0f, 0.0f,
                        0.0f);

      ggml_tensor* kc = session.k_cache[i];
      ggml_tensor* vc = session.v_cache[i];
      ggml_build_forward_expand(
          gf, ggml_cpy(ctx, k,
                       ggml_view_3d(ctx, kc, hd, n_head_kv_, n_tokens, kc->nb[1],
                                    kc->nb[2], session.n_past * kc->nb[2])));
      ggml_build_forward_expand(
          gf, ggml_cpy(ctx, v,
                       ggml_view_3d(ctx, vc, hd, n_head_kv_, n_tokens, vc->nb[1],
                                    vc->nb[2], session.n_past * vc->nb[2])));

      ggml_tensor* k_all = ggml_view_3d(ctx, kc, hd, n_head_kv_, n_kv, kc->nb[1],
                                        kc->nb[2], 0);
      ggml_tensor* v_all = ggml_view_3d(ctx, vc, hd, n_head_kv_, n_kv, vc->nb[1],
                                        vc->nb[2], 0);

      ggml_tensor* kq = ggml_mul_mat(ctx, ggml_permute(ctx, k_all, 0, 2, 1, 3),
                                     ggml_permute(ctx, q, 0, 2, 1, 3));
      kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
      ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v_all, 1, 2, 0, 3));
      ggml_tensor* kqv = ggml_mul_mat(ctx, vt, kq);
      ggml_tensor* attn = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
      attn = ggml_mul_mat(ctx, L.wo, ggml_reshape_2d(ctx, attn, n_embd, n_tokens));
      x = ggml_add(ctx, x, attn);

      cur = ggml_mul(ctx, ggml_rms_norm(ctx, x, rms_eps_), L.ffn_norm);
      ggml_tensor* gate = ggml_silu(ctx, ggml_mul_mat(ctx, L.w_gate, cur));
      ggml_tensor* up = ggml_mul_mat(ctx, L.w_up, cur);
      x = ggml_add(ctx, x, ggml_mul_mat(ctx, L.w_down, ggml_mul(ctx, gate, up)));
    }
    uint32_t rows = n_tokens;
    if (output_ != nullptr) {
      if (last_row_only) {
        x = ggml_cont(ctx, ggml_view_2d(ctx, x, n_embd, 1, x->nb[1],
                                        (n_tokens - 1) * x->nb[1]));
        rows = 1;
      }
      x = ggml_mul(ctx, ggml_rms_norm(ctx, x, rms_eps_), out_norm_);
      x = ggml_mul_mat(ctx, output_, x);
    }
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    if (!ggml_gallocr_alloc_graph(galloc_, gf)) {
      fail(ErrorKind::kResourceExhausted, "cannot allocate the compute graph");
    }

    if (tok_embd_ != nullptr) {
      ggml_backend_tensor_set(inp, input.tokens.data(), 0,
                              sizeof(int32_t) * n_tokens);
    } else {
      ggml_backend_tensor_set(inp, input.hidden.data(), 0,
                              sizeof(float) * input.hidden.size());
    }
    std::vector<int32_t> positions(n_tokens);
    for (uint32_t t = 0; t < n_tokens; ++t) {
      positions[t] = static_cast<int32_t>(session.n_past + t);
    }
    ggml_backend_tensor_set(pos, positions.data(), 0,
                            sizeof(int32_t) * n_tokens);
    std::vector<float> mask_data(static_cast<size_t>(n_kv) * n_tokens);
    for (uint32_t t = 0; t < n_tokens; ++t) {
      for (int j = 0; j < n_kv; ++j) {
        mask_data[static_cast<size_t>(t) * n_kv + j] =
            j <= static_cast<int>(session.n_past + t) ? 0.0f : -INFINITY;
      }
    }
    ggml_backend_tensor_set(mask, mask_data.data(), 0,
                            sizeof(float) * mask_data.size());

    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
      fail(ErrorKind::kInternal, "the compute graph failed");
    }

    StageOutput result;
    result.rows = rows;
    result.is_logits = output_ != nullptr;
    result.data.resize(ggml_nelements(x));
    ggml_backend_tensor_get(x, result.data.data(), 0, ggml_nbytes(x));
    session.n_past += n_tokens;
    return result;
  }

 private:
  uint32_t count_tokens(const StageInput& input) const {
    if (tok_embd_ != nullptr) {
      if (!input.hidden.empty()) {
        fail(ErrorKind::kInvalidArgument,
             "this shard holds the embedding and takes token ids");
      }
      if (input.tokens.empty()) {
        fail(ErrorKind::kInvalidArgument, "no token ids");
      }
      for (int32_t token : input.tokens) {
        if (token < 0 || static_cast<uint32_t>(token) >= model_.n_vocab) {
          fail(ErrorKind::kInvalidArgument,
               "token id " + std::to_string(token) + " is outside the vocabulary");
        }
      }
      return static_cast<uint32_t>(input.tokens.size());
    }
    if (!input.tokens.empty()) {
      fail(ErrorKind::kInvalidArgument,
           "this shard takes hidden states, not token ids");
    }
    if (input.hidden.empty() || input.hidden.size() % model_.n_embd != 0) {
      fail(ErrorKind::kInvalidArgument,
           "hidden states must be whole rows of " +
               std::to_string(model_.n_embd) + " floats");
    }
    return static_cast<uint32_t>(input.hidden.size() / model_.n_embd);
  }

  void open_backend() {
    if (spec_.device == DeviceKind::kCpu) {
      backend_ = ggml_backend_cpu_init();
      if (backend_ == nullptr) {
        fail(ErrorKind::kInternal, "cannot initialise the CPU backend");
      }
      ggml_backend_cpu_set_n_threads(backend_, spec_.threads);
    } else {
      const std::string wanted = accelerator_name(spec_.device);
      for (size_t i = 0; i < ggml_backend_dev_count() && backend_ == nullptr; ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        if (wanted != ggml_backend_reg_name(ggml_backend_dev_backend_reg(device))) {
          continue;
        }
        backend_ = ggml_backend_dev_init(device, nullptr);
      }
      if (backend_ == nullptr) {
        fail(ErrorKind::kFailedPrecondition,
             std::string("this worker has no ") + device_kind_name(spec_.device) +
                 " device in ggml");
      }
    }
    shard_.backend = ggml_backend_name(backend_);
  }

  void read_config() {
    gguf_init_params params{true, &meta_};
    gguf_ = gguf_init_from_file(spec_.model_path.c_str(), params);
    if (gguf_ == nullptr) {
      fail(ErrorKind::kNotFound, "cannot read the model file");
    }
    int64_t arch = gguf_find_key(gguf_, "general.architecture");
    std::string name = (arch >= 0 && gguf_get_kv_type(gguf_, arch) == GGUF_TYPE_STRING)
                           ? gguf_get_val_str(gguf_, arch)
                           : "";
    if (name != "llama") {
      fail(ErrorKind::kInvalidArgument,
           "unsupported architecture '" + name + "', only llama is supported");
    }

    model_.n_layer = get_u32(gguf_, "llama.block_count");
    model_.n_embd = get_u32(gguf_, "llama.embedding_length");
    n_ff_ = get_u32(gguf_, "llama.feed_forward_length");
    n_head_ = get_u32(gguf_, "llama.attention.head_count");
    n_head_kv_ = get_u32(gguf_, "llama.attention.head_count_kv");
    model_.n_ctx_train = get_u32(gguf_, "llama.context_length");
    rope_base_ = get_f32(gguf_, "llama.rope.freq_base");
    rms_eps_ = get_f32(gguf_, "llama.attention.layer_norm_rms_epsilon");

    if (model_.n_layer == 0 || model_.n_embd == 0 || n_ff_ == 0 || n_head_ == 0 ||
        n_head_kv_ == 0 || model_.n_embd % n_head_ != 0 || n_head_ % n_head_kv_ != 0) {
      fail(ErrorKind::kInvalidArgument, "the model's hyperparameters are inconsistent");
    }
    head_dim_ = model_.n_embd / n_head_;
    if (head_dim_ % 2 != 0) {
      fail(ErrorKind::kInvalidArgument, "the head size must be even for RoPE");
    }

    ggml_tensor* embd = ggml_get_tensor(meta_, "token_embd.weight");
    if (embd == nullptr) {
      fail(ErrorKind::kInvalidArgument, "the model has no token_embd.weight");
    }
    model_.n_vocab = static_cast<uint32_t>(embd->ne[1]);

    shard_.layer_begin = spec_.layer_begin;
    shard_.layer_end = spec_.layer_end == 0 ? model_.n_layer : spec_.layer_end;
    if (shard_.layer_begin >= shard_.layer_end || shard_.layer_end > model_.n_layer) {
      fail(ErrorKind::kInvalidArgument,
           "layers [" + std::to_string(shard_.layer_begin) + ", " +
               std::to_string(shard_.layer_end) + ") are outside the model's " +
               std::to_string(model_.n_layer));
    }
    context_length_ = spec_.context_length != 0
                          ? spec_.context_length
                          : std::min(kDefaultContextLength, model_.n_ctx_train);
    if (context_length_ == 0 || context_length_ > model_.n_ctx_train) {
      fail(ErrorKind::kInvalidArgument,
           "the context length must be between 1 and the trained " +
               std::to_string(model_.n_ctx_train));
    }
    shard_.context_length = context_length_;
    shard_.has_embedding = shard_.layer_begin == 0;
    shard_.has_head = shard_.layer_end == model_.n_layer;
  }

  ggml_tensor* declare(const std::string& name, std::initializer_list<int64_t> ne,
                       bool must_be_f32 = false) {
    ggml_tensor* source = ggml_get_tensor(meta_, name.c_str());
    if (source == nullptr) {
      fail(ErrorKind::kInvalidArgument, "the model has no tensor " + name);
    }
    expect_shape(source, ne);
    if (must_be_f32) {
      expect_f32(source);
    }
    ggml_tensor* tensor = ggml_dup_tensor(ctx_w_, source);
    ggml_set_name(tensor, name.c_str());
    return tensor;
  }

  void load_weights() {
    const int64_t n_embd = model_.n_embd;
    const int64_t n_kv = static_cast<int64_t>(n_head_kv_) * head_dim_;
    const size_t n_layers = shard_.layer_end - shard_.layer_begin;
    ggml_init_params params{ggml_tensor_overhead() * (9 * n_layers + 3), nullptr, true};
    ctx_w_ = ggml_init(params);
    if (ctx_w_ == nullptr) {
      fail(ErrorKind::kInternal, "cannot create the weights context");
    }

    if (shard_.has_embedding) {
      tok_embd_ = declare("token_embd.weight", {n_embd, model_.n_vocab});
    }
    for (uint32_t l = shard_.layer_begin; l < shard_.layer_end; ++l) {
      std::string p = "blk." + std::to_string(l) + ".";
      Layer layer;
      layer.attn_norm = declare(p + "attn_norm.weight", {n_embd}, true);
      layer.wq = declare(p + "attn_q.weight", {n_embd, n_embd});
      layer.wk = declare(p + "attn_k.weight", {n_embd, n_kv});
      layer.wv = declare(p + "attn_v.weight", {n_embd, n_kv});
      layer.wo = declare(p + "attn_output.weight", {n_embd, n_embd});
      layer.ffn_norm = declare(p + "ffn_norm.weight", {n_embd}, true);
      layer.w_gate = declare(p + "ffn_gate.weight", {n_embd, n_ff_});
      layer.w_up = declare(p + "ffn_up.weight", {n_embd, n_ff_});
      layer.w_down = declare(p + "ffn_down.weight", {n_ff_, n_embd});
      layers_.push_back(layer);
    }
    if (shard_.has_head) {
      out_norm_ = declare("output_norm.weight", {n_embd}, true);
      output_ = declare("output.weight", {n_embd, model_.n_vocab});
    }

    buf_w_ = ggml_backend_alloc_ctx_tensors(ctx_w_, backend_);
    if (buf_w_ == nullptr) {
      fail(ErrorKind::kResourceExhausted, "not enough memory for this shard's weights");
    }

    FILE* file = std::fopen(spec_.model_path.c_str(), "rb");
    if (file == nullptr) {
      fail(ErrorKind::kNotFound, "cannot open the model file");
    }
    std::unique_ptr<FILE, int (*)(FILE*)> closer(file, &std::fclose);
    std::vector<uint8_t> staging;
    const size_t data_offset = gguf_get_data_offset(gguf_);
    for (ggml_tensor* t = ggml_get_first_tensor(ctx_w_); t != nullptr;
         t = ggml_get_next_tensor(ctx_w_, t)) {
      int64_t id = gguf_find_tensor(gguf_, ggml_get_name(t));
      const size_t offset = data_offset + gguf_get_tensor_offset(gguf_, id);
      const size_t size = ggml_nbytes(t);
      staging.resize(size);
      if (std::fseek(file, static_cast<long>(offset), SEEK_SET) != 0 ||
          std::fread(staging.data(), 1, size, file) != size) {
        fail(ErrorKind::kInvalidArgument,
             std::string("the model file is truncated at ") + ggml_get_name(t));
      }
      ggml_backend_tensor_set(t, staging.data(), 0, size);
      shard_.weight_bytes += size;
    }
  }

  std::unique_ptr<Session> make_session() {
    auto session = std::make_unique<Session>();
    const size_t n = 2 * layers_.size();
    ggml_init_params params{ggml_tensor_overhead() * n, nullptr, true};
    session->context = ggml_init(params);
    if (session->context == nullptr) {
      fail(ErrorKind::kInternal, "cannot create a KV cache context");
    }
    for (size_t i = 0; i < layers_.size(); ++i) {
      session->k_cache.push_back(ggml_new_tensor_3d(
          session->context, GGML_TYPE_F32, head_dim_, n_head_kv_, context_length_));
      session->v_cache.push_back(ggml_new_tensor_3d(
          session->context, GGML_TYPE_F32, head_dim_, n_head_kv_, context_length_));
    }
    session->buffer = ggml_backend_alloc_ctx_tensors(session->context, backend_);
    if (session->buffer == nullptr) {
      fail(ErrorKind::kResourceExhausted, "not enough memory for a KV cache");
    }
    ggml_backend_buffer_clear(session->buffer, 0);
    return session;
  }

  ShardSpec spec_;
  ModelInfo model_;
  ShardInfo shard_;
  uint32_t n_ff_ = 0;
  uint32_t n_head_ = 0;
  uint32_t n_head_kv_ = 0;
  uint32_t head_dim_ = 0;
  uint32_t context_length_ = 0;
  float rope_base_ = 10000.0f;
  float rms_eps_ = 1e-5f;

  gguf_context* gguf_ = nullptr;
  ggml_context* meta_ = nullptr;
  ggml_backend_t backend_ = nullptr;

  ggml_context* ctx_w_ = nullptr;
  ggml_backend_buffer_t buf_w_ = nullptr;
  ggml_tensor* tok_embd_ = nullptr;
  ggml_tensor* out_norm_ = nullptr;
  ggml_tensor* output_ = nullptr;
  std::vector<Layer> layers_;

  ggml_gallocr_t galloc_ = nullptr;
  std::map<std::string, std::unique_ptr<Session>> sessions_;
};

}  // namespace

std::unique_ptr<Backend> load_ggml_backend(const ShardSpec& spec) {
  auto backend = std::make_unique<GgmlBackend>(spec);
  backend->load();
  return backend;
}

}  // namespace shardline
