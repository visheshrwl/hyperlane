#include "tiny_model.h"

#include <filesystem>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <vector>

#include "ggml.h"
#include "gguf.h"

namespace shardline::testing {

std::string make_temp_dir() {
  std::random_device random;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto path = std::filesystem::temp_directory_path() /
                ("shardline-test-" + std::to_string(random()));
    if (std::filesystem::create_directory(path)) {
      return path.string();
    }
  }
  throw std::runtime_error("cannot create a temporary directory");
}

std::string write_tiny_model(const std::string& dir, const std::string& name,
                             const TinyModelOptions& o) {
  std::mt19937 rng(o.seed);
  std::normal_distribution<float> noise(0.0f, 1.0f);
  const int64_t n_embd = o.n_embd;
  const int64_t n_kv = static_cast<int64_t>(o.n_head_kv) * (o.n_embd / o.n_head);

  struct Spec {
    std::string name;
    int64_t ne0;
    int64_t ne1;  // 0 for a vector
    bool norm;
  };
  std::vector<Spec> specs;
  specs.push_back({"token_embd.weight", n_embd, o.n_vocab, false});
  for (uint32_t l = 0; l < o.n_layer; ++l) {
    std::string p = "blk." + std::to_string(l) + ".";
    specs.push_back({p + "attn_norm.weight", n_embd, 0, true});
    specs.push_back({p + "attn_q.weight", n_embd, n_embd, false});
    specs.push_back({p + "attn_k.weight", n_embd, n_kv, false});
    specs.push_back({p + "attn_v.weight", n_embd, n_kv, false});
    specs.push_back({p + "attn_output.weight", n_embd, n_embd, false});
    specs.push_back({p + "ffn_norm.weight", n_embd, 0, true});
    specs.push_back({p + "ffn_gate.weight", n_embd, o.n_ff, false});
    specs.push_back({p + "ffn_up.weight", n_embd, o.n_ff, false});
    int64_t down_in = o.wrong_ffn_down_shape ? o.n_ff + 1 : o.n_ff;
    specs.push_back({p + "ffn_down.weight", down_in, n_embd, false});
  }
  specs.push_back({"output_norm.weight", n_embd, 0, true});
  specs.push_back({"output.weight", n_embd, o.n_vocab, false});

  size_t bytes = 1 << 16;
  for (const Spec& s : specs) {
    bytes += sizeof(float) * s.ne0 * (s.ne1 == 0 ? 1 : s.ne1) +
             ggml_tensor_overhead() + 64;
  }
  ggml_init_params params{bytes, nullptr, false};
  ggml_context* ctx = ggml_init(params);
  gguf_context* gguf = gguf_init_empty();
  if (ctx == nullptr || gguf == nullptr) {
    throw std::runtime_error("cannot create the tiny model");
  }

  gguf_set_val_str(gguf, "general.architecture", "llama");
  gguf_set_val_u32(gguf, "llama.block_count", o.n_layer);
  gguf_set_val_u32(gguf, "llama.context_length", o.n_ctx);
  gguf_set_val_u32(gguf, "llama.embedding_length", o.n_embd);
  gguf_set_val_u32(gguf, "llama.feed_forward_length", o.n_ff);
  gguf_set_val_u32(gguf, "llama.attention.head_count", o.n_head);
  gguf_set_val_u32(gguf, "llama.attention.head_count_kv", o.n_head_kv);
  gguf_set_val_f32(gguf, "llama.rope.freq_base", 10000.0f);
  gguf_set_val_f32(gguf, "llama.attention.layer_norm_rms_epsilon", 1e-5f);

  for (const Spec& s : specs) {
    ggml_tensor* t = s.ne1 == 0 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.ne0)
                                : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.ne0, s.ne1);
    ggml_set_name(t, s.name.c_str());
    float* values = static_cast<float*>(t->data);
    const float spread = s.norm ? 0.1f : 0.25f;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
      values[i] = (s.norm ? 1.0f : 0.0f) + spread * noise(rng);
    }
    gguf_add_tensor(gguf, t);
  }

  std::string path = dir + "/" + name;
  bool written = gguf_write_to_file(gguf, path.c_str(), false);
  gguf_free(gguf);
  ggml_free(ctx);
  if (!written) {
    throw std::runtime_error("cannot write " + path);
  }
  return path;
}

}  // namespace shardline::testing
