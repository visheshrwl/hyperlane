#include "stage.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

namespace shardline::spike {

namespace {

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

uint32_t get_u32(gguf_context* gguf, const std::string& key) {
  int64_t id = gguf_find_key(gguf, key.c_str());
  if (id < 0) throw std::runtime_error("missing GGUF key " + key);
  return gguf_get_val_u32(gguf, id);
}

float get_f32(gguf_context* gguf, const std::string& key) {
  int64_t id = gguf_find_key(gguf, key.c_str());
  if (id < 0) throw std::runtime_error("missing GGUF key " + key);
  return gguf_get_val_f32(gguf, id);
}

}  // namespace

struct Stage::Impl {
  StageSpec spec;
  ModelConfig cfg;
  std::string backend_name;

  gguf_context* gguf = nullptr;
  ggml_context* meta = nullptr;
  ggml_backend_t backend = nullptr;

  ggml_context* ctx_w = nullptr;
  ggml_backend_buffer_t buf_w = nullptr;
  ggml_tensor* tok_embd = nullptr;
  ggml_tensor* out_norm = nullptr;
  ggml_tensor* output = nullptr;
  std::vector<Layer> layers;
  size_t weight_bytes = 0;

  ggml_context* ctx_kv = nullptr;
  ggml_backend_buffer_t buf_kv = nullptr;
  std::vector<ggml_tensor*> k_cache;
  std::vector<ggml_tensor*> v_cache;
  int n_past = 0;

  ggml_gallocr_t galloc = nullptr;

  ~Impl() {
    if (galloc) ggml_gallocr_free(galloc);
    if (buf_kv) ggml_backend_buffer_free(buf_kv);
    if (ctx_kv) ggml_free(ctx_kv);
    if (buf_w) ggml_backend_buffer_free(buf_w);
    if (ctx_w) ggml_free(ctx_w);
    if (backend) ggml_backend_free(backend);
    if (meta) ggml_free(meta);
    if (gguf) gguf_free(gguf);
  }

  void open_backend() {
    if (spec.backend == "cpu") {
      backend = ggml_backend_cpu_init();
      ggml_backend_cpu_set_n_threads(backend, spec.threads);
    } else if (spec.backend == "gpu") {
      ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
      if (dev == nullptr) throw std::runtime_error("no GPU backend in this build");
      backend = ggml_backend_dev_init(dev, nullptr);
    } else {
      throw std::runtime_error("unknown backend " + spec.backend);
    }
    if (backend == nullptr) throw std::runtime_error("cannot initialise backend");
    backend_name = ggml_backend_name(backend);
  }

  void read_config() {
    gguf_init_params params{true, &meta};
    gguf = gguf_init_from_file(spec.model_path.c_str(), params);
    if (gguf == nullptr) throw std::runtime_error("cannot read " + spec.model_path);
    int64_t arch = gguf_find_key(gguf, "general.architecture");
    std::string a = arch < 0 ? "" : gguf_get_val_str(gguf, arch);
    if (a != "llama") throw std::runtime_error("unsupported architecture '" + a + "'");

    cfg.n_layer = get_u32(gguf, "llama.block_count");
    cfg.n_embd = get_u32(gguf, "llama.embedding_length");
    cfg.n_ff = get_u32(gguf, "llama.feed_forward_length");
    cfg.n_head = get_u32(gguf, "llama.attention.head_count");
    cfg.n_head_kv = get_u32(gguf, "llama.attention.head_count_kv");
    cfg.head_dim = cfg.n_embd / cfg.n_head;
    cfg.n_ctx_train = get_u32(gguf, "llama.context_length");
    cfg.rope_base = get_f32(gguf, "llama.rope.freq_base");
    cfg.rms_eps = get_f32(gguf, "llama.attention.layer_norm_rms_epsilon");
    ggml_tensor* embd = ggml_get_tensor(meta, "token_embd.weight");
    if (embd == nullptr) throw std::runtime_error("missing token_embd.weight");
    cfg.n_vocab = static_cast<uint32_t>(embd->ne[1]);

    if (spec.end < 0) spec.end = static_cast<int>(cfg.n_layer);
    if (spec.begin < 0 || spec.begin >= spec.end ||
        spec.end > static_cast<int>(cfg.n_layer)) {
      throw std::runtime_error("invalid layer range");
    }
  }

  ggml_tensor* declare(const std::string& name) {
    ggml_tensor* source = ggml_get_tensor(meta, name.c_str());
    if (source == nullptr) throw std::runtime_error("missing tensor " + name);
    ggml_tensor* t = ggml_dup_tensor(ctx_w, source);
    ggml_set_name(t, name.c_str());
    return t;
  }

  void load_weights() {
    size_t n_tensors = 9 * (spec.end - spec.begin) + 3;
    ggml_init_params params{ggml_tensor_overhead() * n_tensors, nullptr, true};
    ctx_w = ggml_init(params);

    if (spec.begin == 0) tok_embd = declare("token_embd.weight");
    for (int l = spec.begin; l < spec.end; ++l) {
      std::string p = "blk." + std::to_string(l) + ".";
      Layer layer;
      layer.attn_norm = declare(p + "attn_norm.weight");
      layer.wq = declare(p + "attn_q.weight");
      layer.wk = declare(p + "attn_k.weight");
      layer.wv = declare(p + "attn_v.weight");
      layer.wo = declare(p + "attn_output.weight");
      layer.ffn_norm = declare(p + "ffn_norm.weight");
      layer.w_gate = declare(p + "ffn_gate.weight");
      layer.w_up = declare(p + "ffn_up.weight");
      layer.w_down = declare(p + "ffn_down.weight");
      layers.push_back(layer);
    }
    if (spec.end == static_cast<int>(cfg.n_layer)) {
      out_norm = declare("output_norm.weight");
      output = declare("output.weight");
    }

    buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    if (buf_w == nullptr) throw std::runtime_error("cannot allocate weights");

    FILE* file = std::fopen(spec.model_path.c_str(), "rb");
    if (file == nullptr) throw std::runtime_error("cannot open " + spec.model_path);
    std::vector<uint8_t> staging;
    size_t data_offset = gguf_get_data_offset(gguf);
    for (ggml_tensor* t = ggml_get_first_tensor(ctx_w); t != nullptr;
         t = ggml_get_next_tensor(ctx_w, t)) {
      int64_t id = gguf_find_tensor(gguf, ggml_get_name(t));
      size_t offset = data_offset + gguf_get_tensor_offset(gguf, id);
      size_t size = ggml_nbytes(t);
      staging.resize(size);
      if (std::fseek(file, static_cast<long>(offset), SEEK_SET) != 0 ||
          std::fread(staging.data(), 1, size, file) != size) {
        std::fclose(file);
        throw std::runtime_error(std::string("short read for ") + ggml_get_name(t));
      }
      ggml_backend_tensor_set(t, staging.data(), 0, size);
      weight_bytes += size;
    }
    std::fclose(file);
  }

  void allocate_kv() {
    size_t n = 2 * layers.size();
    ggml_init_params params{ggml_tensor_overhead() * n, nullptr, true};
    ctx_kv = ggml_init(params);
    for (size_t i = 0; i < layers.size(); ++i) {
      k_cache.push_back(ggml_new_tensor_3d(ctx_kv, GGML_TYPE_F32, cfg.head_dim,
                                           cfg.n_head_kv, spec.n_ctx));
      v_cache.push_back(ggml_new_tensor_3d(ctx_kv, GGML_TYPE_F32, cfg.head_dim,
                                           cfg.n_head_kv, spec.n_ctx));
    }
    buf_kv = ggml_backend_alloc_ctx_tensors(ctx_kv, backend);
    ggml_backend_buffer_clear(buf_kv, 0);
  }

  std::vector<float> forward(const std::vector<int32_t>& tokens,
                             const std::vector<float>& hidden, int n_tokens) {
    if (n_past + n_tokens > spec.n_ctx) throw std::runtime_error("context full");
    const int n_kv = n_past + n_tokens;
    const int hd = static_cast<int>(cfg.head_dim);

    constexpr size_t kMaxNodes = 8192;
    ggml_init_params params{
        ggml_tensor_overhead() * kMaxNodes + ggml_graph_overhead_custom(kMaxNodes, false),
        nullptr, true};
    ggml_context* ctx = ggml_init(params);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, kMaxNodes, false);

    ggml_tensor* inp = nullptr;
    ggml_tensor* x = nullptr;
    if (tok_embd != nullptr) {
      inp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
      ggml_set_input(inp);
      x = ggml_get_rows(ctx, tok_embd, inp);
    } else {
      inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.n_embd, n_tokens);
      ggml_set_input(inp);
      x = inp;
    }
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(pos);
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_tokens);
    ggml_set_input(mask);

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    for (size_t i = 0; i < layers.size(); ++i) {
      const Layer& L = layers[i];
      ggml_tensor* cur = ggml_mul(ctx, ggml_rms_norm(ctx, x, cfg.rms_eps), L.attn_norm);

      ggml_tensor* q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wq, cur), hd, cfg.n_head, n_tokens);
      ggml_tensor* k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wk, cur), hd, cfg.n_head_kv, n_tokens);
      ggml_tensor* v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wv, cur), hd, cfg.n_head_kv, n_tokens);
      q = ggml_rope_ext(ctx, q, pos, nullptr, hd, GGML_ROPE_TYPE_NORMAL, cfg.n_ctx_train,
                        cfg.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
      k = ggml_rope_ext(ctx, k, pos, nullptr, hd, GGML_ROPE_TYPE_NORMAL, cfg.n_ctx_train,
                        cfg.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

      ggml_tensor* kc = k_cache[i];
      ggml_tensor* vc = v_cache[i];
      ggml_build_forward_expand(
          gf, ggml_cpy(ctx, k, ggml_view_3d(ctx, kc, hd, cfg.n_head_kv, n_tokens, kc->nb[1],
                                            kc->nb[2], n_past * kc->nb[2])));
      ggml_build_forward_expand(
          gf, ggml_cpy(ctx, v, ggml_view_3d(ctx, vc, hd, cfg.n_head_kv, n_tokens, vc->nb[1],
                                            vc->nb[2], n_past * vc->nb[2])));

      ggml_tensor* k_all = ggml_view_3d(ctx, kc, hd, cfg.n_head_kv, n_kv, kc->nb[1], kc->nb[2], 0);
      ggml_tensor* v_all = ggml_view_3d(ctx, vc, hd, cfg.n_head_kv, n_kv, vc->nb[1], vc->nb[2], 0);

      ggml_tensor* kq = ggml_mul_mat(ctx, ggml_permute(ctx, k_all, 0, 2, 1, 3),
                                     ggml_permute(ctx, q, 0, 2, 1, 3));
      kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
      ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v_all, 1, 2, 0, 3));
      ggml_tensor* kqv = ggml_mul_mat(ctx, vt, kq);
      ggml_tensor* attn = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
      attn = ggml_mul_mat(ctx, L.wo, ggml_reshape_2d(ctx, attn, cfg.n_embd, n_tokens));
      x = ggml_add(ctx, x, attn);

      cur = ggml_mul(ctx, ggml_rms_norm(ctx, x, cfg.rms_eps), L.ffn_norm);
      ggml_tensor* gate = ggml_silu(ctx, ggml_mul_mat(ctx, L.w_gate, cur));
      ggml_tensor* up = ggml_mul_mat(ctx, L.w_up, cur);
      x = ggml_add(ctx, x, ggml_mul_mat(ctx, L.w_down, ggml_mul(ctx, gate, up)));
    }
    if (output != nullptr) {
      x = ggml_mul(ctx, ggml_rms_norm(ctx, x, cfg.rms_eps), out_norm);
      x = ggml_mul_mat(ctx, output, x);
    }
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
      ggml_free(ctx);
      throw std::runtime_error("cannot allocate compute graph");
    }

    if (tok_embd != nullptr) {
      ggml_backend_tensor_set(inp, tokens.data(), 0, sizeof(int32_t) * n_tokens);
    } else {
      ggml_backend_tensor_set(inp, hidden.data(), 0, sizeof(float) * n_tokens * cfg.n_embd);
    }
    std::vector<int32_t> positions(n_tokens);
    for (int t = 0; t < n_tokens; ++t) positions[t] = n_past + t;
    ggml_backend_tensor_set(pos, positions.data(), 0, sizeof(int32_t) * n_tokens);
    std::vector<float> mask_data(static_cast<size_t>(n_kv) * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
      for (int j = 0; j < n_kv; ++j) {
        mask_data[static_cast<size_t>(t) * n_kv + j] = j <= n_past + t ? 0.0f : -INFINITY;
      }
    }
    ggml_backend_tensor_set(mask, mask_data.data(), 0, sizeof(float) * mask_data.size());

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
      ggml_free(ctx);
      throw std::runtime_error("graph compute failed");
    }

    std::vector<float> out(ggml_nelements(x));
    ggml_backend_tensor_get(x, out.data(), 0, ggml_nbytes(x));
    ggml_free(ctx);
    n_past += n_tokens;
    return out;
  }
};

Stage::Stage(const StageSpec& spec) : impl_(std::make_unique<Impl>()) {
  impl_->spec = spec;
  impl_->open_backend();
  impl_->read_config();
  impl_->load_weights();
  impl_->allocate_kv();
  impl_->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->backend));
}

Stage::~Stage() = default;

const ModelConfig& Stage::config() const { return impl_->cfg; }
const std::string& Stage::backend_name() const { return impl_->backend_name; }
bool Stage::has_embedding() const { return impl_->tok_embd != nullptr; }
bool Stage::has_head() const { return impl_->output != nullptr; }
size_t Stage::weight_bytes() const { return impl_->weight_bytes; }
int Stage::n_past() const { return impl_->n_past; }
void Stage::reset() { impl_->n_past = 0; }

std::vector<float> Stage::forward(const std::vector<int32_t>& tokens,
                                  const std::vector<float>& hidden, int n_tokens) {
  return impl_->forward(tokens, hidden, n_tokens);
}

}  // namespace shardline::spike
