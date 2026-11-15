"""Greedy generation across a pipeline of workers.

The coordinator relays every hop: it sends the tokens to the first worker, passes the
hidden states it gets back to the next one, and reads the greedy token from the last.
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import grpc

from shardline.client import CHANNEL_OPTIONS
from shardline.v1 import worker_pb2, worker_pb2_grpc

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """The pipeline cannot run as asked: a bad split, or a worker that stopped answering."""


@dataclass(frozen=True)
class Generation:
    tokens: list[int]
    # float32 little-endian rows of [rows, vocab], in the order the model produced them:
    # one per prompt token, then one per decoded token. None unless logits were asked for.
    logits: bytes | None
    rows: int
    vocab: int


class _Stream:
    """One Forward stream, used one request at a time."""

    def __init__(self, stub: worker_pb2_grpc.WorkerServiceStub, timeout: float):
        self._requests: queue.SimpleQueue[worker_pb2.ForwardRequest | None] = queue.SimpleQueue()
        self._call = stub.Forward(self._pending())
        self._timeout = timeout
        self._timed_out = False

    def _pending(self) -> Iterator[worker_pb2.ForwardRequest]:
        while (request := self._requests.get()) is not None:
            yield request

    def _give_up(self) -> None:
        self._timed_out = True
        self._call.cancel()

    def call(self, request: worker_pb2.ForwardRequest) -> worker_pb2.ForwardResponse:
        self._requests.put(request)
        timer = threading.Timer(self._timeout, self._give_up)
        timer.start()
        try:
            return next(self._call)
        except StopIteration:
            raise PipelineError("the worker ended the stream") from None
        except grpc.RpcError as error:
            if self._timed_out:
                raise PipelineError(f"the worker did not answer in {self._timeout:g} s") from error
            raise
        finally:
            timer.cancel()

    def close(self) -> None:
        self._requests.put(None)
        self._call.cancel()


class Pipeline:
    """A model split into consecutive layer ranges, one per worker.

    `cuts` are the layers where a new worker starts: two workers and cuts=(11,) give layers
    [0, 11) to the first and the rest to the second.
    """

    def __init__(
        self,
        targets: Sequence[str],
        model_sha256: str,
        cuts: Sequence[int] = (),
        *,
        device: int = worker_pb2.DEVICE_KIND_UNSPECIFIED,
        context_length: int = 0,
        step_timeout: float = 120.0,
        load_timeout: float = 600.0,
    ):
        if not targets:
            raise ValueError("a pipeline needs at least one worker")
        if len(cuts) != len(targets) - 1:
            raise ValueError(
                f"{len(targets)} workers need {len(targets) - 1} cuts, got {len(cuts)}"
            )
        if any(cut <= 0 for cut in cuts) or list(cuts) != sorted(set(cuts)):
            raise ValueError("cuts must be positive and strictly increasing")
        self._targets = list(targets)
        self._model_sha256 = model_sha256
        self._cuts = list(cuts)
        self._device = device
        self._context_length = context_length
        self._step_timeout = step_timeout
        self._load_timeout = load_timeout
        self._channels: list[grpc.Channel] = []
        self._stubs: list[worker_pb2_grpc.WorkerServiceStub] = []
        self._streams: list[_Stream] = []
        self.model: worker_pb2.ModelInfo | None = None

    def load(self) -> list[worker_pb2.LoadShardResponse]:
        """Loads each worker's layers and opens a stream to each."""
        self.close()
        for target in self._targets:
            channel = grpc.insecure_channel(target, options=CHANNEL_OPTIONS)
            self._channels.append(channel)
            self._stubs.append(worker_pb2_grpc.WorkerServiceStub(channel))

        loaded = []
        for index, stub in enumerate(self._stubs):
            begin = 0 if index == 0 else self._cuts[index - 1]
            end = self._cuts[index] if index < len(self._cuts) else 0
            request = worker_pb2.LoadShardRequest(
                model_sha256=self._model_sha256,
                layer_begin=begin,
                layer_end=end,
                context_length=self._context_length,
                device=self._device,
            )
            loaded.append(stub.LoadShard(request, timeout=self._load_timeout))
        self._check(loaded)

        self.model = loaded[0].model
        self._streams = [_Stream(stub, self._step_timeout) for stub in self._stubs]
        return loaded

    def _check(self, loaded: list[worker_pb2.LoadShardResponse]) -> None:
        first, last = loaded[0].shard, loaded[-1].shard
        if not first.has_embedding or not last.has_head:
            raise PipelineError("the first worker must hold the embedding and the last the head")
        for before, after in zip(loaded, loaded[1:], strict=False):
            if before.shard.layer_end != after.shard.layer_begin:
                raise PipelineError("the workers' layer ranges are not consecutive")
            if before.model != after.model:
                raise PipelineError("the workers loaded different models")
        if any(
            shard.shard.has_embedding != (index == 0)
            or shard.shard.has_head != (index == len(loaded) - 1)
            for index, shard in enumerate(loaded)
        ):
            raise PipelineError(
                "only the first worker holds the embedding and only the last the head"
            )

    def generate(
        self, prompt: Sequence[int], max_new_tokens: int, *, logits: bool = False
    ) -> Generation:
        """Greedily generates up to max_new_tokens tokens that follow the prompt."""
        if not self._streams or self.model is None:
            raise PipelineError("load the pipeline first")
        if not prompt or max_new_tokens < 1:
            raise ValueError("a prompt and at least one new token are needed")

        session = uuid.uuid4().hex
        tokens: list[int] = []
        rows: list[bytes] = []
        position = 0
        step_input = list(prompt)
        try:
            for _ in range(max_new_tokens):
                response = self._step(session, position, step_input, logits)
                position += len(step_input)
                tokens.append(response.next_token)
                if logits:
                    rows.append(response.logits.data)
                step_input = [response.next_token]
        finally:
            self._close_session(session)

        vocab = self.model.n_vocab
        data = b"".join(rows) if logits else None
        return Generation(tokens, data, len(data) // (4 * vocab) if data else 0, vocab)

    def _step(
        self, session: str, position: int, tokens: list[int], logits: bool
    ) -> worker_pb2.ForwardResponse:
        request = worker_pb2.ForwardRequest(
            session_id=session,
            position=position,
            tokens=worker_pb2.TokenIds(ids=tokens),
            return_logits=logits,
        )
        for index, stream in enumerate(self._streams):
            response = stream.call(request)
            if index == len(self._streams) - 1:
                return response
            request = worker_pb2.ForwardRequest(
                session_id=session,
                position=position,
                hidden=response.hidden,
                return_logits=logits,
            )
        raise AssertionError("unreachable: a pipeline has at least one stream")

    def _close_session(self, session: str) -> None:
        for stub in self._stubs:
            try:
                stub.CloseSession(worker_pb2.CloseSessionRequest(session_id=session), timeout=10)
            except grpc.RpcError as error:
                # Best effort: the session is already gone if its stream or the worker ended,
                # and this runs while another error may be on its way to the caller.
                logger.debug("could not close session %s: %s", session, error)

    def close(self) -> None:
        for stream in self._streams:
            stream.close()
        for channel in self._channels:
            channel.close()
        self._streams, self._stubs, self._channels = [], [], []

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
