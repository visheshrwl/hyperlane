"""gRPC client for a single shardline worker."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import grpc

from shardline.v1 import worker_pb2, worker_pb2_grpc


def first_reachable(targets: Sequence[str], timeout: float = 1.0) -> str | None:
    """The first target, in the order given, that accepts a gRPC connection in time.

    Every target is tried at once, so the wait is one timeout however many there are.
    """
    if not targets:
        return None

    def accepts_connection(target: str) -> bool:
        channel = grpc.insecure_channel(target)
        try:
            grpc.channel_ready_future(channel).result(timeout=timeout)
            return True
        except grpc.FutureTimeoutError:
            return False
        finally:
            channel.close()

    with ThreadPoolExecutor(max_workers=len(targets)) as pool:
        reachable = list(pool.map(accepts_connection, targets))
    return next((target for target, ok in zip(targets, reachable, strict=True) if ok), None)


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
