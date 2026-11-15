#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

#include "shardline/backend.h"
#include "shardline/v1/worker.grpc.pb.h"
#include "shardline/worker_node.h"
#include "tiny_model.h"

namespace shardline {
namespace {

const std::string kDigest(64, 'b');
const std::vector<int32_t> kPrompt = {3, 17, 5, 42, 8, 11, 2};

class Node {
 public:
  explicit Node(const std::string& model_dir) {
    WorkerOptions options;
    options.listen_address = "127.0.0.1:0";
    options.model_dir = model_dir;
    options.threads = 2;
    node_ = std::make_unique<WorkerNode>(std::move(options));
    EXPECT_TRUE(node_->start());
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(kMaxMessageBytes);
    args.SetMaxSendMessageSize(kMaxMessageBytes);
    stub_ = v1::WorkerService::NewStub(grpc::CreateCustomChannel(
        "127.0.0.1:" + std::to_string(node_->port()),
        grpc::InsecureChannelCredentials(), args));
  }

  grpc::Status load(uint32_t begin, uint32_t end, v1::LoadShardResponse* response) {
    v1::LoadShardRequest request;
    request.set_model_sha256(kDigest);
    request.set_layer_begin(begin);
    request.set_layer_end(end);
    request.set_context_length(64);
    request.set_device(v1::DEVICE_KIND_CPU);
    grpc::ClientContext context;
    return stub_->LoadShard(&context, request, response);
  }

  v1::WorkerService::Stub& stub() { return *stub_; }

 private:
  std::unique_ptr<WorkerNode> node_;
  std::unique_ptr<v1::WorkerService::Stub> stub_;
};

std::vector<float> floats(const v1::Tensor& tensor) {
  std::vector<float> out(tensor.data().size() / sizeof(float));
  std::memcpy(out.data(), tensor.data().data(), tensor.data().size());
  return out;
}

grpc::Status forward(v1::WorkerService::Stub& stub, const v1::ForwardRequest& request,
                     v1::ForwardResponse* response) {
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(20));
  auto stream = stub.Forward(&context);
  if (!stream->Write(request) || !stream->Read(response)) {
    return stream->Finish();
  }
  stream->WritesDone();
  return stream->Finish();
}

class GgmlServiceTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    dir_ = testing::make_temp_dir();
    std::string made = testing::write_tiny_model(dir_, "source.gguf");
    std::filesystem::copy_file(made, dir_ + "/" + kDigest + ".gguf");
  }
  static void TearDownTestSuite() { std::filesystem::remove_all(dir_); }
  static std::string dir_;
};

std::string GgmlServiceTest::dir_;

TEST_F(GgmlServiceTest, ServesARealShardOverGrpc) {
  Node node(dir_);
  v1::LoadShardResponse loaded;
  ASSERT_TRUE(node.load(0, 0, &loaded).ok());
  EXPECT_EQ(loaded.model().n_layer(), 4u);
  EXPECT_EQ(loaded.model().n_vocab(), 48u);
  EXPECT_TRUE(loaded.shard().has_embedding());
  EXPECT_TRUE(loaded.shard().has_head());
  EXPECT_EQ(loaded.shard().device(), v1::DEVICE_KIND_CPU);
  EXPECT_GT(loaded.shard().weight_bytes(), 0u);

  v1::ForwardRequest request;
  request.set_session_id("s");
  for (int32_t id : kPrompt) request.mutable_tokens()->add_ids(id);
  request.set_return_logits(true);
  v1::ForwardResponse response;
  ASSERT_TRUE(forward(node.stub(), request, &response).ok());
  ASSERT_TRUE(response.has_logits());
  EXPECT_EQ(response.logits().shape(0), kPrompt.size());
  EXPECT_EQ(response.logits().shape(1), 48u);

  // The service adds nothing to what the engine computes.
  ShardSpec spec;
  spec.model_path = dir_ + "/" + kDigest + ".gguf";
  spec.context_length = 64;
  spec.threads = 2;
  auto direct = load_ggml_backend(spec);
  direct->open_session("s");
  StageInput input;
  input.tokens = kPrompt;
  input.last_row_only = false;
  StageOutput expected = direct->forward("s", 0, std::move(input));
  EXPECT_EQ(floats(response.logits()), expected.data);

  const float* last = expected.data.data() + (expected.rows - 1) * 48;
  EXPECT_EQ(response.next_token(), std::max_element(last, last + 48) - last);
}

TEST_F(GgmlServiceTest, TwoWorkersGiveTheSameLogitsAsOne) {
  Node whole(dir_);
  Node first(dir_);
  Node second(dir_);
  v1::LoadShardResponse scratch;
  ASSERT_TRUE(whole.load(0, 0, &scratch).ok());
  ASSERT_TRUE(first.load(0, 2, &scratch).ok());
  ASSERT_TRUE(second.load(2, 0, &scratch).ok());

  auto request = [](bool logits) {
    v1::ForwardRequest r;
    r.set_session_id("s");
    for (int32_t id : kPrompt) r.mutable_tokens()->add_ids(id);
    r.set_return_logits(logits);
    return r;
  };

  v1::ForwardResponse one;
  ASSERT_TRUE(forward(whole.stub(), request(true), &one).ok());

  v1::ForwardResponse middle;
  ASSERT_TRUE(forward(first.stub(), request(false), &middle).ok());
  ASSERT_TRUE(middle.has_hidden());
  v1::ForwardRequest onward;
  onward.set_session_id("s");
  *onward.mutable_hidden() = middle.hidden();
  onward.set_return_logits(true);
  v1::ForwardResponse two;
  ASSERT_TRUE(forward(second.stub(), onward, &two).ok());

  EXPECT_EQ(floats(two.logits()), floats(one.logits()))
      << "a pipeline split over two workers is bit-exact with one worker";
  EXPECT_EQ(two.next_token(), one.next_token());
}

}  // namespace
}  // namespace shardline
