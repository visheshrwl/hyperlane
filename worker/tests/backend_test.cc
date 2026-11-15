#include <algorithm>
#include <cmath>
#include <filesystem>

#include <gtest/gtest.h>

#include "shardline/backend.h"
#include "tiny_model.h"

namespace shardline {
namespace {

using testing::TinyModelOptions;
using testing::make_temp_dir;
using testing::write_tiny_model;

const std::vector<int32_t> kPrompt = {3, 17, 5, 42, 8, 11, 2};

template <typename F>
void expect_error(ErrorKind kind, F&& call) {
  try {
    call();
    FAIL() << "expected a BackendError";
  } catch (const BackendError& error) {
    EXPECT_EQ(error.kind(), kind) << error.what();
  }
}

class TinyModelTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    dir_ = make_temp_dir();
    path_ = write_tiny_model(dir_, "tiny.gguf");
  }
  static void TearDownTestSuite() { std::filesystem::remove_all(dir_); }

  static ShardSpec spec(uint32_t begin = 0, uint32_t end = 0) {
    ShardSpec s;
    s.model_path = path_;
    s.layer_begin = begin;
    s.layer_end = end;
    s.context_length = 64;
    s.threads = 2;
    return s;
  }

  static std::unique_ptr<Backend> load(uint32_t begin = 0, uint32_t end = 0) {
    return load_ggml_backend(spec(begin, end));
  }

  static StageInput tokens(std::vector<int32_t> ids, bool last_row_only = false) {
    StageInput input;
    input.tokens = std::move(ids);
    input.last_row_only = last_row_only;
    return input;
  }

  static StageInput hidden(std::vector<float> rows, bool last_row_only = false) {
    StageInput input;
    input.hidden = std::move(rows);
    input.last_row_only = last_row_only;
    return input;
  }

  static int argmax_last_row(const StageOutput& out, uint32_t n_vocab) {
    const float* row = out.data.data() + (out.rows - 1) * n_vocab;
    return static_cast<int>(std::max_element(row, row + n_vocab) - row);
  }

  static std::string dir_;
  static std::string path_;
};

std::string TinyModelTest::dir_;
std::string TinyModelTest::path_;

TEST_F(TinyModelTest, ReportsTheShardItLoaded) {
  auto whole = load();
  EXPECT_EQ(whole->model().n_layer, 4u);
  EXPECT_EQ(whole->model().n_embd, 32u);
  EXPECT_EQ(whole->model().n_vocab, 48u);
  EXPECT_TRUE(whole->shard().has_embedding);
  EXPECT_TRUE(whole->shard().has_head);
  EXPECT_EQ(whole->shard().layer_end, 4u);
  EXPECT_GT(whole->shard().weight_bytes, 0u);

  auto first = load(0, 2);
  auto last = load(2, 4);
  EXPECT_TRUE(first->shard().has_embedding);
  EXPECT_FALSE(first->shard().has_head);
  EXPECT_FALSE(last->shard().has_embedding);
  EXPECT_TRUE(last->shard().has_head);
  EXPECT_EQ(first->shard().weight_bytes + last->shard().weight_bytes -
                whole->shard().weight_bytes,
            0u)
      << "the shards together hold exactly the whole model";
}

TEST_F(TinyModelTest, TwoStagesMatchOneStageBitForBit) {
  auto whole = load();
  auto first = load(0, 2);
  auto last = load(2, 4);
  for (Backend* b : {whole.get(), first.get(), last.get()}) b->open_session("s");

  StageOutput expected = whole->forward("s", 0, tokens(kPrompt));
  StageOutput middle = first->forward("s", 0, tokens(kPrompt));
  ASSERT_FALSE(middle.is_logits);
  StageOutput actual = last->forward("s", 0, hidden(middle.data));
  ASSERT_TRUE(actual.is_logits);
  EXPECT_EQ(actual.rows, kPrompt.size());
  EXPECT_EQ(actual.data, expected.data);

  uint32_t position = kPrompt.size();
  int next = argmax_last_row(expected, 48);
  for (int step = 0; step < 6; ++step, ++position) {
    expected = whole->forward("s", position, tokens({next}));
    middle = first->forward("s", position, tokens({next}));
    actual = last->forward("s", position, hidden(middle.data));
    ASSERT_EQ(actual.data, expected.data) << "decode step " << step;
    next = argmax_last_row(expected, 48);
  }
}

TEST_F(TinyModelTest, TheKvCacheMatchesRecomputingEverything) {
  auto backend = load();
  backend->open_session("whole");
  backend->open_session("incremental");

  StageOutput whole = backend->forward("whole", 0, tokens(kPrompt));
  StageOutput part = backend->forward("incremental", 0, tokens({3, 17, 5}));
  uint32_t position = 3;
  StageOutput last;
  for (size_t i = 3; i < kPrompt.size(); ++i, ++position) {
    last = backend->forward("incremental", position, tokens({kPrompt[i]}));
  }

  const float* want = whole.data.data() + (whole.rows - 1) * 48;
  ASSERT_EQ(last.rows, 1u);
  for (int i = 0; i < 48; ++i) {
    EXPECT_NEAR(last.data[i], want[i], 1e-4) << "logit " << i;
  }
}

TEST_F(TinyModelTest, LastRowOnlyAgreesWithTheFullOutput) {
  auto backend = load();
  backend->open_session("full");
  backend->open_session("short");

  StageOutput full = backend->forward("full", 0, tokens(kPrompt, false));
  StageOutput one = backend->forward("short", 0, tokens(kPrompt, true));

  ASSERT_EQ(full.rows, kPrompt.size());
  ASSERT_EQ(one.rows, 1u);
  ASSERT_EQ(one.data.size(), 48u);
  const float* want = full.data.data() + (full.rows - 1) * 48;
  for (int i = 0; i < 48; ++i) {
    EXPECT_NEAR(one.data[i], want[i], 1e-4) << "logit " << i;
  }
}

TEST_F(TinyModelTest, SessionsDoNotShareState) {
  auto shared = load();
  shared->open_session("a");
  shared->open_session("b");
  auto alone_a = load();
  auto alone_b = load();
  alone_a->open_session("a");
  alone_b->open_session("b");

  const std::vector<int32_t> other = {9, 1, 30, 4};
  StageOutput a1 = shared->forward("a", 0, tokens(kPrompt));
  StageOutput b1 = shared->forward("b", 0, tokens(other));
  StageOutput a2 = shared->forward("a", kPrompt.size(), tokens({6}));
  StageOutput b2 = shared->forward("b", other.size(), tokens({7}));

  EXPECT_EQ(a1.data, alone_a->forward("a", 0, tokens(kPrompt)).data);
  EXPECT_EQ(a2.data, alone_a->forward("a", kPrompt.size(), tokens({6})).data);
  EXPECT_EQ(b1.data, alone_b->forward("b", 0, tokens(other)).data);
  EXPECT_EQ(b2.data, alone_b->forward("b", other.size(), tokens({7})).data);
}

TEST_F(TinyModelTest, ASessionCanBeClosedAndItsIdReused) {
  auto backend = load();
  backend->open_session("s");
  StageOutput first = backend->forward("s", 0, tokens(kPrompt));
  backend->close_session("s");
  EXPECT_FALSE(backend->has_session("s"));

  backend->open_session("s");
  EXPECT_EQ(backend->forward("s", 0, tokens(kPrompt)).data, first.data)
      << "a reopened session starts with an empty cache";
}

TEST_F(TinyModelTest, RejectsRequestsThatWouldCorruptASession) {
  auto first = load(0, 2);
  auto last = load(2, 4);
  first->open_session("s");
  last->open_session("s");
  first->forward("s", 0, tokens({1, 2, 3}));

  expect_error(ErrorKind::kFailedPrecondition,
               [&] { first->forward("s", 0, tokens({4})); });
  expect_error(ErrorKind::kFailedPrecondition,
               [&] { first->forward("s", 9, tokens({4})); });
  expect_error(ErrorKind::kInvalidArgument,
               [&] { first->forward("s", 3, tokens({48})); });
  expect_error(ErrorKind::kInvalidArgument,
               [&] { first->forward("s", 3, tokens({-1})); });
  expect_error(ErrorKind::kInvalidArgument, [&] { first->forward("s", 3, tokens({})); });
  expect_error(ErrorKind::kInvalidArgument,
               [&] { first->forward("s", 3, hidden(std::vector<float>(32, 0.0f))); });
  expect_error(ErrorKind::kInvalidArgument,
               [&] { last->forward("s", 0, tokens({1})); });
  expect_error(ErrorKind::kInvalidArgument,
               [&] { last->forward("s", 0, hidden(std::vector<float>(33, 0.0f))); });
  expect_error(ErrorKind::kInvalidArgument, [&] { last->forward("s", 0, hidden({})); });
  expect_error(ErrorKind::kNotFound, [&] { first->forward("nope", 0, tokens({1})); });

  EXPECT_NO_THROW(first->forward("s", 3, tokens({4})))
      << "a refused request leaves the session where it was";
}

TEST_F(TinyModelTest, RefusesToOverflowTheContext) {
  auto backend = load();
  backend->open_session("s");
  expect_error(ErrorKind::kResourceExhausted,
               [&] { backend->forward("s", 0, tokens(std::vector<int32_t>(65, 1))); });
  EXPECT_NO_THROW(backend->forward("s", 0, tokens(std::vector<int32_t>(64, 1))));
  expect_error(ErrorKind::kResourceExhausted,
               [&] { backend->forward("s", 64, tokens({1})); });
}

TEST_F(TinyModelTest, ManagesSessionLifetimes) {
  ShardSpec s = spec();
  s.max_sessions = 2;
  auto backend = load_ggml_backend(s);
  backend->open_session("a");
  expect_error(ErrorKind::kAlreadyExists, [&] { backend->open_session("a"); });
  backend->open_session("b");
  expect_error(ErrorKind::kResourceExhausted, [&] { backend->open_session("c"); });
  expect_error(ErrorKind::kNotFound, [&] { backend->close_session("c"); });
  backend->close_session("a");
  EXPECT_NO_THROW(backend->open_session("c"));
}

TEST_F(TinyModelTest, RefusesBadLayerRangesAndContexts) {
  expect_error(ErrorKind::kInvalidArgument, [&] { load(2, 2); });
  expect_error(ErrorKind::kInvalidArgument, [&] { load(3, 2); });
  expect_error(ErrorKind::kInvalidArgument, [&] { load(0, 5); });
  expect_error(ErrorKind::kInvalidArgument, [&] { load(4, 0); });

  ShardSpec s = spec();
  s.context_length = 65;
  expect_error(ErrorKind::kInvalidArgument, [&] { load_ggml_backend(s); });
  s.context_length = 0;
  expect_error(ErrorKind::kInvalidArgument, [&] { load_ggml_backend(s); });
}

TEST_F(TinyModelTest, RefusesAMissingOrDamagedModel) {
  ShardSpec missing = spec();
  missing.model_path = dir_ + "/absent.gguf";
  expect_error(ErrorKind::kNotFound, [&] { load_ggml_backend(missing); });

  TinyModelOptions bad;
  bad.wrong_ffn_down_shape = true;
  ShardSpec wrong = spec();
  wrong.model_path = write_tiny_model(dir_, "wrong.gguf", bad);
  expect_error(ErrorKind::kInvalidArgument, [&] { load_ggml_backend(wrong); });

  std::string junk = dir_ + "/junk.gguf";
  FILE* file = std::fopen(junk.c_str(), "wb");
  std::fputs("this is not a model", file);
  std::fclose(file);
  ShardSpec garbage = spec();
  garbage.model_path = junk;
  expect_error(ErrorKind::kNotFound, [&] { load_ggml_backend(garbage); });
}

TEST_F(TinyModelTest, ThreadCountDoesNotChangeTheResult) {
  ShardSpec one = spec();
  one.threads = 1;
  ShardSpec four = spec();
  four.threads = 4;
  auto a = load_ggml_backend(one);
  auto b = load_ggml_backend(four);
  a->open_session("s");
  b->open_session("s");
  EXPECT_EQ(a->forward("s", 0, tokens(kPrompt)).data,
            b->forward("s", 0, tokens(kPrompt)).data);
}

#if defined(SHARDLINE_TEST_METAL)
TEST_F(TinyModelTest, MetalAgreesWithTheCpu) {
  ShardSpec s = spec();
  s.device = DeviceKind::kMetal;
  std::unique_ptr<Backend> metal;
  try {
    metal = load_ggml_backend(s);
  } catch (const BackendError& error) {
    GTEST_SKIP() << "no Metal device: " << error.what();
  }
  auto cpu = load();
  metal->open_session("s");
  cpu->open_session("s");

  StageOutput want = cpu->forward("s", 0, tokens(kPrompt));
  StageOutput got = metal->forward("s", 0, tokens(kPrompt));
  ASSERT_EQ(got.data.size(), want.data.size());
  for (size_t i = 0; i < want.data.size(); ++i) {
    ASSERT_NEAR(got.data[i], want.data[i], 1e-2) << "element " << i;
  }
  EXPECT_EQ(argmax_last_row(got, 48), argmax_last_row(want, 48));
}
#endif

}  // namespace
}  // namespace shardline
