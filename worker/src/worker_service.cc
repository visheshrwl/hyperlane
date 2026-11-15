#include "shardline/worker_service.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <iostream>

#include "shardline/devices.h"
#include "shardline/protocol.h"

namespace shardline {

namespace {

static_assert(std::endian::native == std::endian::little,
              "tensors travel as little-endian float32");

constexpr size_t kMaxSessionIdBytes = 128;
constexpr uint32_t kDefaultContextLength = 512;

v1::DeviceKind to_proto(DeviceKind kind) {
  switch (kind) {
    case DeviceKind::kCpu:
      return v1::DEVICE_KIND_CPU;
    case DeviceKind::kMetal:
      return v1::DEVICE_KIND_METAL;
    case DeviceKind::kCuda:
      return v1::DEVICE_KIND_CUDA;
  }
  return v1::DEVICE_KIND_UNSPECIFIED;
}

grpc::StatusCode to_code(ErrorKind kind) {
  switch (kind) {
    case ErrorKind::kInvalidArgument:
      return grpc::StatusCode::INVALID_ARGUMENT;
    case ErrorKind::kNotFound:
      return grpc::StatusCode::NOT_FOUND;
    case ErrorKind::kAlreadyExists:
      return grpc::StatusCode::ALREADY_EXISTS;
    case ErrorKind::kResourceExhausted:
      return grpc::StatusCode::RESOURCE_EXHAUSTED;
    case ErrorKind::kFailedPrecondition:
      return grpc::StatusCode::FAILED_PRECONDITION;
    case ErrorKind::kUnimplemented:
      return grpc::StatusCode::UNIMPLEMENTED;
    case ErrorKind::kInternal:
      break;
  }
  return grpc::StatusCode::INTERNAL;
}

grpc::Status invalid(const std::string& message) {
  return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, message);
}

// The devices to try, in order. An unspecified request takes the first
// accelerator the worker reports that ggml can use, and falls back to the CPU.
std::vector<DeviceKind> candidates(v1::DeviceKind requested) {
  switch (requested) {
    case v1::DEVICE_KIND_CPU:
      return {DeviceKind::kCpu};
    case v1::DEVICE_KIND_METAL:
      return {DeviceKind::kMetal};
    case v1::DEVICE_KIND_CUDA:
      return {DeviceKind::kCuda};
    default:
      break;
  }
  std::vector<DeviceKind> kinds;
  for (const DeviceInfo& device : probe_devices()) {
    if (device.kind != DeviceKind::kCpu) {
      kinds.push_back(device.kind);
    }
  }
  kinds.push_back(DeviceKind::kCpu);
  return kinds;
}

void set_tensor(v1::Tensor* tensor, const std::vector<float>& data,
                uint32_t rows, uint32_t cols) {
  tensor->set_type(v1::TENSOR_TYPE_F32);
  tensor->add_shape(rows);
  tensor->add_shape(cols);
  tensor->set_data(data.data(), data.size() * sizeof(float));
}

// Checks a [rows, n_embd] float32 tensor and copies it out.
grpc::Status read_hidden(const v1::Tensor& tensor, uint32_t n_embd,
                         std::vector<float>* out) {
  if (tensor.type() != v1::TENSOR_TYPE_F32) {
    return invalid("hidden states must be float32");
  }
  if (tensor.shape_size() != 2 || tensor.shape(0) == 0 ||
      tensor.shape(1) != n_embd) {
    return invalid("hidden states must have shape [n_tokens, " +
                   std::to_string(n_embd) + "]");
  }
  const uint64_t bytes = static_cast<uint64_t>(tensor.shape(0)) * n_embd * sizeof(float);
  if (tensor.data().size() != bytes) {
    return invalid("the hidden-state data does not match its shape");
  }
  out->resize(static_cast<size_t>(tensor.shape(0)) * n_embd);
  std::memcpy(out->data(), tensor.data().data(), bytes);
  return grpc::Status::OK;
}

}  // namespace

const char* worker_version() { return SHARDLINE_VERSION; }

WorkerServiceImpl::WorkerServiceImpl(std::string worker_id,
                                     std::string hostname,
                                     ServiceConfig config)
    : worker_id_(std::move(worker_id)),
      hostname_(std::move(hostname)),
      config_(std::move(config)) {
  if (!config_.backend_factory) {
    config_.backend_factory = load_ggml_backend;
  }
}

grpc::Status WorkerServiceImpl::GetStats(grpc::ServerContext*,
                                         const v1::GetStatsRequest*,
                                         v1::GetStatsResponse* response) {
  response->set_worker_id(worker_id_);
  response->set_hostname(hostname_);
  response->set_protocol(kProtocol);
  response->set_worker_version(worker_version());
  for (const DeviceInfo& device : probe_devices()) {
    v1::DeviceInfo* entry = response->add_devices();
    entry->set_kind(to_proto(device.kind));
    entry->set_name(device.name);
    entry->set_total_memory_bytes(device.total_memory_bytes);
    entry->set_usable_memory_bytes(device.usable_memory_bytes);
  }
  return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::LoadShard(grpc::ServerContext*,
                                          const v1::LoadShardRequest* request,
                                          v1::LoadShardResponse* response) {
  try {
    if (config_.model_dir.empty()) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "this worker has no model directory; start it with --model-dir");
    }
    ShardSpec spec;
    spec.model_path = model_path_for_digest(config_.model_dir, request->model_sha256());
    spec.layer_begin = request->layer_begin();
    spec.layer_end = request->layer_end();
    spec.context_length =
        request->context_length() == 0 ? kDefaultContextLength : request->context_length();
    spec.threads = config_.threads;
    spec.max_sessions = config_.max_sessions;

    std::lock_guard<std::mutex> lock(mutex_);
    // The old shard goes first: a worker holds one, and its memory is needed.
    backend_.reset();
    ++generation_;

    const std::vector<DeviceKind> kinds = candidates(request->device());
    std::unique_ptr<Backend> backend;
    for (size_t i = 0; i < kinds.size() && backend == nullptr; ++i) {
      spec.device = kinds[i];
      try {
        backend = config_.backend_factory(spec);
      } catch (const BackendError& error) {
        bool no_such_device = error.kind() == ErrorKind::kFailedPrecondition;
        if (!no_such_device || i + 1 == kinds.size()) {
          throw;
        }
      }
    }
    if (backend == nullptr) {
      throw BackendError(ErrorKind::kInternal, "no device could run the shard");
    }

    const ModelInfo& model = backend->model();
    const ShardInfo& shard = backend->shard();
    v1::ModelInfo* model_out = response->mutable_model();
    model_out->set_n_layer(model.n_layer);
    model_out->set_n_embd(model.n_embd);
    model_out->set_n_vocab(model.n_vocab);
    model_out->set_n_ctx_train(model.n_ctx_train);
    v1::ShardInfo* shard_out = response->mutable_shard();
    shard_out->set_layer_begin(shard.layer_begin);
    shard_out->set_layer_end(shard.layer_end);
    shard_out->set_has_embedding(shard.has_embedding);
    shard_out->set_has_head(shard.has_head);
    shard_out->set_weight_bytes(shard.weight_bytes);
    shard_out->set_device(to_proto(spec.device));
    shard_out->set_backend(shard.backend);

    std::cerr << "loaded layers [" << shard.layer_begin << ", " << shard.layer_end
              << ") of " << request->model_sha256().substr(0, 12) << " on "
              << shard.backend << "\n";
    backend_ = std::move(backend);
    return grpc::Status::OK;
  } catch (const BackendError& error) {
    return grpc::Status(to_code(error.kind()), error.what());
  } catch (const std::exception& error) {
    return grpc::Status(grpc::StatusCode::INTERNAL, error.what());
  }
}

grpc::Status WorkerServiceImpl::CloseSession(
    grpc::ServerContext*, const v1::CloseSessionRequest* request,
    v1::CloseSessionResponse*) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!backend_ || !backend_->has_session(request->session_id())) {
    return grpc::Status(grpc::StatusCode::NOT_FOUND,
                        "no session " + request->session_id());
  }
  backend_->close_session(request->session_id());
  return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::Forward(
    grpc::ServerContext*,
    grpc::ServerReaderWriter<v1::ForwardResponse, v1::ForwardRequest>* stream) {
  // Sessions belong to the stream that started them, so a coordinator that
  // dies cannot leave KV caches behind.
  std::vector<Owned> owned;
  struct Cleanup {
    WorkerServiceImpl* service;
    std::vector<Owned>* owned;
    ~Cleanup() { service->close_owned(*owned); }
  } cleanup{this, &owned};

  v1::ForwardRequest request;
  while (stream->Read(&request)) {
    v1::ForwardResponse response;
    grpc::Status status = forward_one(request, &response, &owned);
    if (!status.ok()) {
      return status;
    }
    if (!stream->Write(response)) {
      break;
    }
  }
  return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::forward_one(const v1::ForwardRequest& request,
                                            v1::ForwardResponse* response,
                                            std::vector<Owned>* owned) {
  try {
    const std::string& id = request.session_id();
    if (id.empty() || id.size() > kMaxSessionIdBytes) {
      return invalid("a session id is 1 to " + std::to_string(kMaxSessionIdBytes) + " bytes");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!backend_) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "no shard is loaded");
    }
    const ModelInfo& model = backend_->model();

    StageInput input;
    uint32_t n_tokens = 0;
    switch (request.input_case()) {
      case v1::ForwardRequest::kTokens:
        input.tokens.assign(request.tokens().ids().begin(), request.tokens().ids().end());
        n_tokens = static_cast<uint32_t>(input.tokens.size());
        break;
      case v1::ForwardRequest::kHidden: {
        grpc::Status status = read_hidden(request.hidden(), model.n_embd, &input.hidden);
        if (!status.ok()) {
          return status;
        }
        n_tokens = request.hidden().shape(0);
        break;
      }
      case v1::ForwardRequest::INPUT_NOT_SET:
        return invalid("a request carries token ids or hidden states");
    }
    input.last_row_only = !request.return_logits();

    if (!backend_->has_session(id)) {
      if (request.position() != 0) {
        return grpc::Status(grpc::StatusCode::NOT_FOUND,
                            "no session " + id + "; a session starts at position 0");
      }
      backend_->open_session(id);
      owned->emplace_back(id, generation_);
    }

    StageOutput output = backend_->forward(id, request.position(), std::move(input));

    response->set_session_id(id);
    response->set_position(request.position() + n_tokens);
    if (!output.is_logits) {
      set_tensor(response->mutable_hidden(), output.data, output.rows, model.n_embd);
      return grpc::Status::OK;
    }
    const float* last = output.data.data() +
                        static_cast<size_t>(output.rows - 1) * model.n_vocab;
    response->set_next_token(
        static_cast<int32_t>(std::max_element(last, last + model.n_vocab) - last));
    if (request.return_logits()) {
      set_tensor(response->mutable_logits(), output.data, output.rows, model.n_vocab);
    }
    return grpc::Status::OK;
  } catch (const BackendError& error) {
    return grpc::Status(to_code(error.kind()), error.what());
  } catch (const std::exception& error) {
    return grpc::Status(grpc::StatusCode::INTERNAL, error.what());
  }
}

void WorkerServiceImpl::close_owned(const std::vector<Owned>& owned) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!backend_) {
    return;
  }
  for (const auto& [id, generation] : owned) {
    if (generation == generation_ && backend_->has_session(id)) {
      backend_->close_session(id);
    }
  }
}

}  // namespace shardline
