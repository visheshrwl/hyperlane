#include <chrono>
#include <thread>

#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

#include "shardline/v1/worker.grpc.pb.h"
#include "shardline/worker_node.h"

namespace shardline {
namespace {

WorkerOptions loopback_options(const std::string& worker_id = "test-worker") {
  WorkerOptions options;
  options.listen_address = "127.0.0.1:0";
  options.worker_id = worker_id;
  return options;
}

grpc::Status get_stats(int port, v1::GetStatsResponse* response) {
  auto channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(port),
                                     grpc::InsecureChannelCredentials());
  auto stub = v1::WorkerService::NewStub(channel);
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() +
                       std::chrono::seconds(5));
  return stub->GetStats(&context, v1::GetStatsRequest(), response);
}

TEST(WorkerNode, ServesGetStatsOnEphemeralPort) {
  WorkerNode node(loopback_options());
  ASSERT_TRUE(node.start());
  ASSERT_GT(node.port(), 0);

  v1::GetStatsResponse response;
  grpc::Status status = get_stats(node.port(), &response);
  ASSERT_TRUE(status.ok()) << status.error_message();
  EXPECT_EQ(response.worker_id(), "test-worker");
  EXPECT_EQ(response.protocol(), "shardline.v1");
  EXPECT_FALSE(response.hostname().empty());
  EXPECT_FALSE(response.worker_version().empty());
  ASSERT_GT(response.devices_size(), 0);
  EXPECT_EQ(response.devices(response.devices_size() - 1).kind(),
            v1::DEVICE_KIND_CPU);
  for (const auto& device : response.devices()) {
    EXPECT_GT(device.total_memory_bytes(), 0u);
    EXPECT_LE(device.usable_memory_bytes(), device.total_memory_bytes());
  }
}

TEST(WorkerNode, ShutdownReleasesWaitingThread) {
  WorkerNode node(loopback_options());
  ASSERT_TRUE(node.start());
  std::thread waiter([&] { node.wait(); });
  node.shutdown();
  waiter.join();
}

TEST(WorkerNode, DestructorStopsRunningServer) {
  int port = 0;
  {
    WorkerNode node(loopback_options());
    ASSERT_TRUE(node.start());
    port = node.port();
  }
  v1::GetStatsResponse response;
  EXPECT_FALSE(get_stats(port, &response).ok());
}

TEST(WorkerNode, RefusesAPortThatIsAlreadyInUse) {
  WorkerNode first(loopback_options("first"));
  ASSERT_TRUE(first.start());

  WorkerOptions options = loopback_options("second");
  options.listen_address = "127.0.0.1:" + std::to_string(first.port());
  WorkerNode second(options);
  EXPECT_FALSE(second.start());
}

TEST(WorkerNode, DefaultIdIsDerivedFromHostname) {
  WorkerNode a(loopback_options(""));
  WorkerNode b(loopback_options(""));
  EXPECT_FALSE(a.worker_id().empty());
  EXPECT_NE(a.worker_id(), b.worker_id());
}

TEST(LoopbackAddress, Classification) {
  EXPECT_TRUE(is_loopback_address("127.0.0.1:50051"));
  EXPECT_TRUE(is_loopback_address("localhost:1"));
  EXPECT_TRUE(is_loopback_address("[::1]:50051"));
  EXPECT_FALSE(is_loopback_address("0.0.0.0:50051"));
  EXPECT_FALSE(is_loopback_address("[::]:50051"));
  EXPECT_FALSE(is_loopback_address("192.168.1.10:50051"));
}

}  // namespace
}  // namespace shardline
