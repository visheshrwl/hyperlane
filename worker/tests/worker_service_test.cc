#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

#include "shardline/backend.h"
#include "shardline/v1/worker.grpc.pb.h"
#include "shardline/worker_node.h"

namespace shardline {
namespace {

const std::string kDigest(64, 'a');
constexpr uint32_t kLayers = 4;
constexpr uint32_t kEmbd = 4;
constexpr uint32_t kVocab = 10;

// A model with no weights. Each shard adds its number of layers to every
// value, so the shards chained together turn token t into hidden value t + 4,
// and the head's greedy choice for it is (t + 4) % kVocab.
class FakeBackend final : public Backend {
 public:
  explicit FakeBackend(const ShardSpec& spec) {
    model_.n_layer = kLayers;
    model_.n_embd = kEmbd;
    model_.n_vocab = kVocab;
    model_.n_ctx_train = 64;
    shard_.layer_begin = spec.layer_begin;
    shard_.layer_end = spec.layer_end == 0 ? kLayers : spec.layer_end;
    shard_.has_embedding = shard_.layer_begin == 0;
    shard_.has_head = shard_.layer_end == kLayers;
    shard_.weight_bytes = 1234;
    shard_.context_length = spec.context_length != 0 ? spec.context_length : 64;
    shard_.backend = std::string("fake-") + device_kind_name(spec.device);
  }

  const ModelInfo& model() const override { return model_; }
  const ShardInfo& shard() const override { return shard_; }

  void open_session(const std::string& id) override {
    if (!positions_.emplace(id, 0).second) {
      throw BackendError(ErrorKind::kAlreadyExists, "session exists");
    }
  }
  void close_session(const std::string& id) override { positions_.erase(id); }
  bool has_session(const std::string& id) const override {
    return positions_.count(id) != 0;
  }

  StageOutput forward(const std::string& id, uint32_t position,
                      StageInput input) override {
    auto it = positions_.find(id);
    if (it == positions_.end()) {
      throw BackendError(ErrorKind::kNotFound, "no session");
    }
    if (position != it->second) {
      throw BackendError(ErrorKind::kFailedPrecondition, "wrong position");
    }
    const float span = static_cast<float>(shard_.layer_end - shard_.layer_begin);
    std::vector<float> values;
    if (shard_.has_embedding) {
      for (int32_t token : input.tokens) values.push_back(token + span);
    } else {
      for (size_t i = 0; i < input.hidden.size(); i += kEmbd) {
        values.push_back(input.hidden[i] + span);
      }
    }
    it->second += values.size();

    StageOutput out;
    if (!shard_.has_head) {
      out.rows = values.size();
      for (float v : values) out.data.insert(out.data.end(), kEmbd, v);
      return out;
    }
    out.is_logits = true;
    size_t first = input.last_row_only ? values.size() - 1 : 0;
    for (size_t i = first; i < values.size(); ++i) {
      for (uint32_t j = 0; j < kVocab; ++j) {
        out.data.push_back(j == static_cast<uint32_t>(values[i]) % kVocab ? 1.0f : 0.0f);
      }
      ++out.rows;
    }
    return out;
  }

 private:
  ModelInfo model_;
  ShardInfo shard_;
  std::map<std::string, uint32_t> positions_;
};

struct Loaded {
  std::vector<ShardSpec> specs;
};

BackendFactory fake_factory(std::shared_ptr<Loaded> loaded) {
  return [loaded](const ShardSpec& spec) -> std::unique_ptr<Backend> {
    loaded->specs.push_back(spec);
    return std::make_unique<FakeBackend>(spec);
  };
}

class Worker {
 public:
  explicit Worker(std::string model_dir = "/models", BackendFactory factory = nullptr)
      : loaded_(std::make_shared<Loaded>()) {
    WorkerOptions options;
    options.listen_address = "127.0.0.1:0";
    options.worker_id = "service-test";
    options.model_dir = std::move(model_dir);
    options.backend_factory = factory ? factory : fake_factory(loaded_);
    node_ = std::make_unique<WorkerNode>(std::move(options));
    EXPECT_TRUE(node_->start());

    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(kMaxMessageBytes);
    args.SetMaxSendMessageSize(kMaxMessageBytes);
    stub_ = v1::WorkerService::NewStub(grpc::CreateCustomChannel(
        "127.0.0.1:" + std::to_string(node_->port()),
        grpc::InsecureChannelCredentials(), args));
  }

  v1::WorkerService::Stub& stub() { return *stub_; }
  const std::vector<ShardSpec>& specs() const { return loaded_->specs; }

  grpc::Status load(uint32_t begin, uint32_t end,
                    v1::LoadShardResponse* response = nullptr,
                    v1::DeviceKind device = v1::DEVICE_KIND_CPU,
                    const std::string& digest = kDigest) {
    v1::LoadShardRequest request;
    request.set_model_sha256(digest);
    request.set_layer_begin(begin);
    request.set_layer_end(end);
    request.set_device(device);
    v1::LoadShardResponse scratch;
    grpc::ClientContext context;
    return stub_->LoadShard(&context, request, response ? response : &scratch);
  }

  grpc::Status close(const std::string& session) {
    v1::CloseSessionRequest request;
    request.set_session_id(session);
    v1::CloseSessionResponse response;
    grpc::ClientContext context;
    return stub_->CloseSession(&context, request, &response);
  }

 private:
  std::shared_ptr<Loaded> loaded_;
  std::unique_ptr<WorkerNode> node_;
  std::unique_ptr<v1::WorkerService::Stub> stub_;
};

// One Forward stream, used one request at a time.
class Stream {
 public:
  explicit Stream(v1::WorkerService::Stub& stub) {
    context_.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(20));
    stream_ = stub.Forward(&context_);
  }

  // Sends a request and waits for the answer. When the server ends the stream
  // instead, returns the status it ended with.
  grpc::Status call(const v1::ForwardRequest& request, v1::ForwardResponse* response) {
    if (!stream_->Write(request) || !stream_->Read(response)) {
      return stream_->Finish();
    }
    return grpc::Status::OK;
  }

  grpc::Status finish() {
    stream_->WritesDone();
    return stream_->Finish();
  }

 private:
  grpc::ClientContext context_;
  std::unique_ptr<grpc::ClientReaderWriter<v1::ForwardRequest, v1::ForwardResponse>> stream_;
};

v1::ForwardRequest with_tokens(const std::string& session, uint32_t position,
                               std::vector<int32_t> ids, bool return_logits = false) {
  v1::ForwardRequest request;
  request.set_session_id(session);
  request.set_position(position);
  for (int32_t id : ids) request.mutable_tokens()->add_ids(id);
  request.set_return_logits(return_logits);
  return request;
}

v1::ForwardRequest with_hidden(const std::string& session, uint32_t position,
                               const v1::Tensor& hidden, bool return_logits = false) {
  v1::ForwardRequest request;
  request.set_session_id(session);
  request.set_position(position);
  *request.mutable_hidden() = hidden;
  request.set_return_logits(return_logits);
  return request;
}

v1::Tensor hidden_tensor(uint32_t rows, uint32_t cols, float value) {
  v1::Tensor tensor;
  tensor.set_type(v1::TENSOR_TYPE_F32);
  tensor.add_shape(rows);
  tensor.add_shape(cols);
  std::vector<float> data(static_cast<size_t>(rows) * cols, value);
  tensor.set_data(data.data(), data.size() * sizeof(float));
  return tensor;
}

std::vector<float> floats(const v1::Tensor& tensor) {
  std::vector<float> out(tensor.data().size() / sizeof(float));
  std::memcpy(out.data(), tensor.data().data(), tensor.data().size());
  return out;
}

TEST(WorkerService, LoadsAShardByDigest) {
  Worker worker("/models");
  v1::LoadShardResponse response;
  ASSERT_TRUE(worker.load(0, 2, &response).ok());

  ASSERT_EQ(worker.specs().size(), 1u);
  EXPECT_EQ(worker.specs()[0].model_path, "/models/" + kDigest + ".gguf");
  EXPECT_EQ(worker.specs()[0].layer_begin, 0u);
  EXPECT_EQ(worker.specs()[0].layer_end, 2u);
  EXPECT_EQ(worker.specs()[0].context_length, 0u) << "the backend resolves the default";
  EXPECT_EQ(worker.specs()[0].device, DeviceKind::kCpu);

  EXPECT_EQ(response.model().n_layer(), kLayers);
  EXPECT_EQ(response.model().n_embd(), kEmbd);
  EXPECT_EQ(response.model().n_vocab(), kVocab);
  EXPECT_EQ(response.shard().layer_begin(), 0u);
  EXPECT_EQ(response.shard().layer_end(), 2u);
  EXPECT_TRUE(response.shard().has_embedding());
  EXPECT_FALSE(response.shard().has_head());
  EXPECT_EQ(response.shard().weight_bytes(), 1234u);
  EXPECT_EQ(response.shard().context_length(), 64u);
  EXPECT_EQ(response.shard().device(), v1::DEVICE_KIND_CPU);
  EXPECT_EQ(response.shard().backend(), "fake-cpu");
}

TEST(WorkerService, LayerEndZeroMeansThroughTheLastLayer) {
  Worker worker;
  v1::LoadShardResponse response;
  ASSERT_TRUE(worker.load(2, 0, &response).ok());
  EXPECT_EQ(response.shard().layer_end(), kLayers);
  EXPECT_TRUE(response.shard().has_head());
  EXPECT_FALSE(response.shard().has_embedding());
}

TEST(WorkerService, ARequestCannotNameAPath) {
  Worker worker;
  for (const std::string& bad : {std::string("../../etc/passwd"), std::string(),
                                 std::string(63, 'a'), std::string(64, 'A'),
                                 "../" + std::string(61, 'a')}) {
    grpc::Status status = worker.load(0, 0, nullptr, v1::DEVICE_KIND_CPU, bad);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT) << bad;
  }
  EXPECT_TRUE(worker.specs().empty()) << "no backend was asked to open anything";
}

TEST(WorkerService, RefusesToLoadWithoutAModelDirectory) {
  Worker worker("");
  EXPECT_EQ(worker.load(0, 0).error_code(), grpc::StatusCode::FAILED_PRECONDITION);
}

TEST(WorkerService, FallsBackToTheCpuWhenNoAcceleratorCanRunTheShard) {
  auto loaded = std::make_shared<Loaded>();
  Worker worker("/models", [loaded](const ShardSpec& spec) -> std::unique_ptr<Backend> {
    loaded->specs.push_back(spec);
    if (spec.device != DeviceKind::kCpu) {
      throw BackendError(ErrorKind::kFailedPrecondition, "no such device");
    }
    return std::make_unique<FakeBackend>(spec);
  });

  v1::LoadShardResponse response;
  ASSERT_TRUE(worker.load(0, 0, &response, v1::DEVICE_KIND_UNSPECIFIED).ok());
  EXPECT_EQ(response.shard().device(), v1::DEVICE_KIND_CPU);
  EXPECT_EQ(loaded->specs.back().device, DeviceKind::kCpu);

  EXPECT_EQ(worker.load(0, 0, nullptr, v1::DEVICE_KIND_METAL).error_code(),
            grpc::StatusCode::FAILED_PRECONDITION)
      << "an explicit device is never swapped for another";
  EXPECT_EQ(worker.load(0, 0, nullptr, v1::DEVICE_KIND_CUDA).error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);
}

TEST(WorkerService, MapsLoadFailuresToStatusCodes) {
  Worker missing("/models", [](const ShardSpec&) -> std::unique_ptr<Backend> {
    throw BackendError(ErrorKind::kNotFound, "cannot read the model file");
  });
  EXPECT_EQ(missing.load(0, 0).error_code(), grpc::StatusCode::NOT_FOUND);

  Worker broken("/models", [](const ShardSpec&) -> std::unique_ptr<Backend> {
    throw std::runtime_error("boom");
  });
  grpc::Status status = broken.load(0, 0);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
  EXPECT_EQ(status.error_message(), "boom");
}

TEST(WorkerService, ForwardNeedsAShard) {
  Worker worker;
  Stream stream(worker.stub());
  v1::ForwardResponse response;
  EXPECT_EQ(stream.call(with_tokens("s", 0, {1}), &response).error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);
}

TEST(WorkerService, TwoShardsChainedGiveTheSameAnswerAsOne) {
  Worker first;
  Worker second;
  ASSERT_TRUE(first.load(0, 2).ok());
  ASSERT_TRUE(second.load(2, 0).ok());
  Stream a(first.stub());
  Stream b(second.stub());

  v1::ForwardResponse r1;
  v1::ForwardResponse r2;
  ASSERT_TRUE(a.call(with_tokens("s", 0, {1, 2, 3}), &r1).ok());
  EXPECT_EQ(r1.position(), 3u);
  ASSERT_TRUE(r1.has_hidden());
  EXPECT_FALSE(r1.has_next_token());
  ASSERT_EQ(r1.hidden().shape_size(), 2);
  EXPECT_EQ(r1.hidden().shape(0), 3u);
  EXPECT_EQ(r1.hidden().shape(1), kEmbd);

  ASSERT_TRUE(b.call(with_hidden("s", 0, r1.hidden()), &r2).ok());
  EXPECT_EQ(r2.position(), 3u);
  ASSERT_TRUE(r2.has_next_token());
  EXPECT_EQ(r2.next_token(), (3 + 4) % static_cast<int>(kVocab));
  EXPECT_FALSE(r2.has_logits()) << "logits only when asked for";

  int next = r2.next_token();
  uint32_t position = 3;
  for (int step = 0; step < 4; ++step, ++position) {
    ASSERT_TRUE(a.call(with_tokens("s", position, {next}), &r1).ok());
    ASSERT_TRUE(b.call(with_hidden("s", position, r1.hidden()), &r2).ok());
    EXPECT_EQ(r2.next_token(), (next + 4) % static_cast<int>(kVocab));
    next = r2.next_token();
  }
  EXPECT_TRUE(a.finish().ok());
  EXPECT_TRUE(b.finish().ok());
}

TEST(WorkerService, ReturnsLogitsWhenAsked) {
  Worker worker;
  ASSERT_TRUE(worker.load(0, 0).ok());
  Stream stream(worker.stub());

  v1::ForwardResponse response;
  ASSERT_TRUE(stream.call(with_tokens("s", 0, {1, 2, 3}, true), &response).ok());
  ASSERT_TRUE(response.has_logits());
  ASSERT_EQ(response.logits().shape_size(), 2);
  EXPECT_EQ(response.logits().shape(0), 3u) << "one row per input token";
  EXPECT_EQ(response.logits().shape(1), kVocab);
  std::vector<float> logits = floats(response.logits());
  ASSERT_EQ(logits.size(), 3u * kVocab);
  EXPECT_EQ(logits[0 * kVocab + 5], 1.0f);
  EXPECT_EQ(logits[2 * kVocab + 7], 1.0f);
  EXPECT_EQ(response.next_token(), 7);

  ASSERT_TRUE(stream.call(with_tokens("s", 3, {9}), &response).ok());
  EXPECT_FALSE(response.has_logits());
  EXPECT_EQ(response.next_token(), 3);
}

TEST(WorkerService, AWrongPositionEndsTheStreamAndFreesItsSessions) {
  Worker worker;
  ASSERT_TRUE(worker.load(0, 0).ok());
  v1::ForwardResponse response;
  {
    Stream stream(worker.stub());
    ASSERT_TRUE(stream.call(with_tokens("s", 0, {1, 2}), &response).ok());
    EXPECT_EQ(stream.call(with_tokens("s", 0, {3}), &response).error_code(),
              grpc::StatusCode::FAILED_PRECONDITION);
  }
  EXPECT_EQ(worker.close("s").error_code(), grpc::StatusCode::NOT_FOUND)
      << "the session died with its stream";

  Stream again(worker.stub());
  EXPECT_TRUE(again.call(with_tokens("s", 0, {1, 2}), &response).ok());
}

TEST(WorkerService, ASessionLivesUntilItsStreamEnds) {
  Worker worker;
  ASSERT_TRUE(worker.load(0, 0).ok());
  v1::ForwardResponse response;
  Stream stream(worker.stub());
  ASSERT_TRUE(stream.call(with_tokens("s", 0, {1}), &response).ok());
  ASSERT_TRUE(stream.finish().ok());
  EXPECT_EQ(worker.close("s").error_code(), grpc::StatusCode::NOT_FOUND);

  Stream next(worker.stub());
  EXPECT_EQ(next.call(with_tokens("s", 1, {1}), &response).error_code(),
            grpc::StatusCode::NOT_FOUND)
      << "a session starts at position 0";
}

TEST(WorkerService, CloseSessionFreesItEarly) {
  Worker worker;
  ASSERT_TRUE(worker.load(0, 0).ok());
  v1::ForwardResponse response;
  Stream stream(worker.stub());
  ASSERT_TRUE(stream.call(with_tokens("s", 0, {1}), &response).ok());

  EXPECT_TRUE(worker.close("s").ok());
  EXPECT_EQ(worker.close("s").error_code(), grpc::StatusCode::NOT_FOUND);
  EXPECT_EQ(stream.call(with_tokens("s", 1, {1}), &response).error_code(),
            grpc::StatusCode::NOT_FOUND);
}

TEST(WorkerService, AnOldStreamEndingDoesNotCloseASessionOnTheNewShard) {
  Worker worker;
  ASSERT_TRUE(worker.load(0, 0).ok());
  v1::ForwardResponse response;
  Stream old_stream(worker.stub());
  ASSERT_TRUE(old_stream.call(with_tokens("s", 0, {1}), &response).ok());

  // The new shard has a session of the same name, started by another stream.
  ASSERT_TRUE(worker.load(0, 0).ok());
  Stream new_stream(worker.stub());
  ASSERT_TRUE(new_stream.call(with_tokens("s", 0, {2}), &response).ok());

  ASSERT_TRUE(old_stream.finish().ok());
  EXPECT_TRUE(new_stream.call(with_tokens("s", 1, {2}), &response).ok())
      << "the old stream's cleanup left the new shard's session alone";
}

TEST(WorkerService, ReplacingTheShardClosesTheOldSessions) {
  Worker worker;
  ASSERT_TRUE(worker.load(0, 0).ok());
  v1::ForwardResponse response;
  Stream stream(worker.stub());
  ASSERT_TRUE(stream.call(with_tokens("s", 0, {1}), &response).ok());

  ASSERT_TRUE(worker.load(0, 2).ok());
  EXPECT_EQ(worker.close("s").error_code(), grpc::StatusCode::NOT_FOUND);
}

TEST(WorkerService, RefusesMalformedRequests) {
  Worker worker;
  ASSERT_TRUE(worker.load(2, 0).ok());
  v1::ForwardResponse response;

  auto code = [&](const v1::ForwardRequest& request) {
    Stream stream(worker.stub());
    return stream.call(request, &response).error_code();
  };

  v1::ForwardRequest none;
  none.set_session_id("s");
  EXPECT_EQ(code(none), grpc::StatusCode::INVALID_ARGUMENT) << "no input";

  EXPECT_EQ(code(with_hidden("", 0, hidden_tensor(1, kEmbd, 0))),
            grpc::StatusCode::INVALID_ARGUMENT)
      << "empty session id";
  EXPECT_EQ(code(with_hidden(std::string(129, 'x'), 0, hidden_tensor(1, kEmbd, 0))),
            grpc::StatusCode::INVALID_ARGUMENT)
      << "session id too long";

  v1::Tensor wrong_type = hidden_tensor(1, kEmbd, 0);
  wrong_type.set_type(v1::TENSOR_TYPE_UNSPECIFIED);
  EXPECT_EQ(code(with_hidden("s", 0, wrong_type)), grpc::StatusCode::INVALID_ARGUMENT);

  EXPECT_EQ(code(with_hidden("s", 0, hidden_tensor(1, kEmbd + 1, 0))),
            grpc::StatusCode::INVALID_ARGUMENT)
      << "wrong width";
  EXPECT_EQ(code(with_hidden("s", 0, hidden_tensor(0, kEmbd, 0))),
            grpc::StatusCode::INVALID_ARGUMENT)
      << "no rows";

  v1::Tensor short_data = hidden_tensor(2, kEmbd, 0);
  short_data.set_data(std::string(sizeof(float) * kEmbd, '\0'));
  EXPECT_EQ(code(with_hidden("s", 0, short_data)), grpc::StatusCode::INVALID_ARGUMENT)
      << "data shorter than the shape says";

  v1::Tensor flat = hidden_tensor(1, kEmbd, 0);
  flat.clear_shape();
  flat.add_shape(kEmbd);
  EXPECT_EQ(code(with_hidden("s", 0, flat)), grpc::StatusCode::INVALID_ARGUMENT)
      << "one dimension";
}

TEST(WorkerService, AcceptsMessagesLargerThanTheGrpcDefault) {
  Worker worker;
  ASSERT_TRUE(worker.load(2, 0).ok());
  const uint32_t rows = 600000;  // 9.6 MB, over gRPC's 4 MiB default
  Stream stream(worker.stub());
  v1::ForwardResponse response;
  ASSERT_TRUE(stream.call(with_hidden("s", 0, hidden_tensor(rows, kEmbd, 1.0f)),
                          &response)
                  .ok());
  EXPECT_EQ(response.position(), rows);
  EXPECT_EQ(response.next_token(), 3);
}

TEST(WorkerService, ReportsTheDevicesAsBefore) {
  Worker worker;
  grpc::ClientContext context;
  v1::GetStatsResponse stats;
  ASSERT_TRUE(worker.stub().GetStats(&context, v1::GetStatsRequest(), &stats).ok());
  EXPECT_EQ(stats.worker_id(), "service-test");
  EXPECT_GT(stats.devices_size(), 0);
}

}  // namespace
}  // namespace shardline
