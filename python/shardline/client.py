"""gRPC client for a single shardline worker."""

from __future__ import annotations

import grpc

from shardline.v1 import worker_pb2, worker_pb2_grpc


class WorkerClient:
    def __init__(self, target: str, timeout: float = 5.0):
        self._channel = grpc.insecure_channel(target)
        self._stub = worker_pb2_grpc.WorkerServiceStub(self._channel)
        self._timeout = timeout

    def get_stats(self) -> worker_pb2.GetStatsResponse:
        return self._stub.GetStats(worker_pb2.GetStatsRequest(), timeout=self._timeout)

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> WorkerClient:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
