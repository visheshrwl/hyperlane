// Spike harness: chains Stage objects in one process and runs greedy decoding.
//
//   stage_probe --model M.gguf --tokens 1,2,3 --gen 32 [--cuts 11] [--backends gpu,cpu]
//               [--threads 8] [--ctx 512] [--logits out.bin]
//
// --cuts splits the layers into stages (e.g. "11" gives [0,11) and [11,22)).
// --logits writes every logits row produced: one per prompt position, then
// one per decode step, as int32 rows, int32 cols, then float32 data.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "stage.h"

using shardline::spike::Stage;
using shardline::spike::StageSpec;

namespace {

std::vector<std::string> split(const std::string& text, char sep) {
  std::vector<std::string> parts;
  std::stringstream stream(text);
  std::string part;
  while (std::getline(stream, part, sep)) {
    if (!part.empty()) parts.push_back(part);
  }
  return parts;
}

double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int argmax(const float* row, int n) {
  return static_cast<int>(std::max_element(row, row + n) - row);
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, tokens_arg, cuts_arg, backends_arg = "cpu", logits_path;
  int gen = 16, threads = 4, n_ctx = 512;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string key = argv[i];
    std::string value = argv[i + 1];
    if (key == "--model") model = value;
    else if (key == "--tokens") tokens_arg = value;
    else if (key == "--gen") gen = std::stoi(value);
    else if (key == "--cuts") cuts_arg = value;
    else if (key == "--backends") backends_arg = value;
    else if (key == "--threads") threads = std::stoi(value);
    else if (key == "--ctx") n_ctx = std::stoi(value);
    else if (key == "--logits") logits_path = value;
    else {
      std::fprintf(stderr, "unknown option %s\n", key.c_str());
      return 2;
    }
  }
  if (model.empty() || tokens_arg.empty()) {
    std::fprintf(stderr, "--model and --tokens are required\n");
    return 2;
  }

  std::vector<int32_t> prompt;
  for (const auto& t : split(tokens_arg, ',')) prompt.push_back(std::stoi(t));
  std::vector<int> cuts;
  for (const auto& c : split(cuts_arg, ',')) cuts.push_back(std::stoi(c));
  std::vector<std::string> backends = split(backends_arg, ',');

  std::vector<std::unique_ptr<Stage>> stages;
  double load_start = now_ms();
  int begin = 0;
  for (size_t s = 0; s <= cuts.size(); ++s) {
    StageSpec spec;
    spec.model_path = model;
    spec.backend = backends[std::min(s, backends.size() - 1)];
    spec.begin = begin;
    spec.end = s < cuts.size() ? cuts[s] : -1;
    spec.n_ctx = n_ctx;
    spec.threads = threads;
    stages.push_back(std::make_unique<Stage>(spec));
    begin = spec.end;
  }
  double load_ms = now_ms() - load_start;
  const int n_vocab = static_cast<int>(stages.back()->config().n_vocab);

  std::vector<double> stage_ms(stages.size(), 0.0);
  auto run_chain = [&](const std::vector<int32_t>& tokens, bool timed) {
    std::vector<float> hidden;
    for (size_t s = 0; s < stages.size(); ++s) {
      double start = now_ms();
      hidden = stages[s]->forward(tokens, hidden, static_cast<int>(tokens.size()));
      if (timed) stage_ms[s] += now_ms() - start;
    }
    return hidden;
  };

  std::vector<float> all_logits;
  double prefill_start = now_ms();
  std::vector<float> logits = run_chain(prompt, false);
  double prefill_ms = now_ms() - prefill_start;
  all_logits.insert(all_logits.end(), logits.begin(), logits.end());

  std::vector<int32_t> generated;
  int next = argmax(logits.data() + (prompt.size() - 1) * n_vocab, n_vocab);
  double decode_start = now_ms();
  for (int step = 0; step < gen; ++step) {
    generated.push_back(next);
    if (step + 1 == gen) break;
    logits = run_chain({next}, true);
    all_logits.insert(all_logits.end(), logits.begin(), logits.end());
    next = argmax(logits.data(), n_vocab);
  }
  double decode_ms = now_ms() - decode_start;
  int decode_steps = std::max(gen - 1, 1);

  std::printf("tokens:");
  for (size_t i = 0; i < generated.size(); ++i) {
    std::printf("%s%d", i == 0 ? " " : ",", generated[i]);
  }
  std::printf("\n");
  std::printf("{\"load_ms\": %.1f, \"prefill_tokens\": %zu, \"prefill_ms\": %.2f, "
              "\"decode_steps\": %d, \"decode_ms_per_token\": %.3f, \"stages\": [",
              load_ms, prompt.size(), prefill_ms, decode_steps, decode_ms / decode_steps);
  int layer = 0;
  for (size_t s = 0; s < stages.size(); ++s) {
    int end = s < cuts.size() ? cuts[s] : static_cast<int>(stages[s]->config().n_layer);
    std::printf("%s{\"backend\": \"%s\", \"layers\": [%d, %d], \"weight_mib\": %.1f, "
                "\"decode_ms_per_token\": %.3f}",
                s == 0 ? "" : ", ", stages[s]->backend_name().c_str(), layer, end,
                stages[s]->weight_bytes() / 1048576.0, stage_ms[s] / decode_steps);
    layer = end;
  }
  std::printf("]}\n");

  if (!logits_path.empty()) {
    FILE* out = std::fopen(logits_path.c_str(), "wb");
    int32_t rows = static_cast<int32_t>(all_logits.size() / n_vocab);
    int32_t cols = n_vocab;
    std::fwrite(&rows, sizeof(rows), 1, out);
    std::fwrite(&cols, sizeof(cols), 1, out);
    std::fwrite(all_logits.data(), sizeof(float), all_logits.size(), out);
    std::fclose(out);
  }
  return 0;
}
