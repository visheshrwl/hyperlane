import struct
import time
from array import array
from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest

from shardline import cli
from shardline.pipeline import Pipeline, PipelineError
from shardline.v1 import worker_pb2, worker_pb2_grpc

LAYERS, EMBD, VOCAB = 4, 4, 10
DIGEST = "c" * 64


def tensor(rows, values):
    data = array("f", [v for value in values for v in [value] * EMBD])
    return worker_pb2.Tensor(
        type=worker_pb2.TENSOR_TYPE_F32, shape=[rows, EMBD], data=data.tobytes()
    )


class FakeWorker(worker_pb2_grpc.WorkerServiceServicer):
    """The toy model of the C++ service tests: every shard adds its number of layers to each
    value, so the shards together turn token t into t + 4 and the head picks (t + 4) % 10."""

    def __init__(self, hang=False, fail=False, lie_about_shard=False):
        self.hang = hang
        self.fail = fail
        self.lie = lie_about_shard
        self.begin = self.end = 0
        self.positions = []
        self.closed = []
        self.loaded = []

    def LoadShard(self, request, context):
        self.loaded.append(request)
        self.begin, self.end = request.layer_begin, request.layer_end or LAYERS
        if self.lie:
            self.begin, self.end = 0, LAYERS
        shard = worker_pb2.ShardInfo(
            layer_begin=self.begin,
            layer_end=self.end,
            has_embedding=self.begin == 0,
            has_head=self.end == LAYERS,
            backend="fake",
        )
        model = worker_pb2.ModelInfo(n_layer=LAYERS, n_embd=EMBD, n_vocab=VOCAB, n_ctx_train=64)
        return worker_pb2.LoadShardResponse(model=model, shard=shard)

    def Forward(self, request_iterator, context):
        for request in request_iterator:
            if self.hang:
                while context.is_active():
                    time.sleep(0.02)
                return
            if self.fail:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "wrong position")
            self.positions.append(request.position)
            span = self.end - self.begin
            if request.HasField("tokens"):
                values = [token + span for token in request.tokens.ids]
            else:
                hidden = array("f")
                hidden.frombytes(request.hidden.data)
                values = [hidden[i] + span for i in range(0, len(hidden), EMBD)]
            response = worker_pb2.ForwardResponse(
                session_id=request.session_id, position=request.position + len(values)
            )
            if self.end != LAYERS:
                response.hidden.CopyFrom(tensor(len(values), values))
            else:
                shown = values if request.return_logits else values[-1:]
                rows = array("f")
                for value in shown:
                    rows.extend(1.0 if j == int(value) % VOCAB else 0.0 for j in range(VOCAB))
                response.next_token = int(values[-1]) % VOCAB
                if request.return_logits:
                    response.logits.CopyFrom(
                        worker_pb2.Tensor(
                            type=worker_pb2.TENSOR_TYPE_F32,
                            shape=[len(shown), VOCAB],
                            data=rows.tobytes(),
                        )
                    )
            yield response

    def CloseSession(self, request, context):
        self.closed.append(request.session_id)
        return worker_pb2.CloseSessionResponse()


@pytest.fixture
def start_worker():
    servers = []

    def start(**options):
        worker = FakeWorker(**options)
        server = grpc.server(ThreadPoolExecutor(max_workers=8))
        worker_pb2_grpc.add_WorkerServiceServicer_to_server(worker, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        servers.append(server)
        return worker, f"127.0.0.1:{port}"

    yield start
    for server in servers:
        server.stop(grace=None)


def test_two_workers_generate_what_one_would(start_worker):
    first, first_target = start_worker()
    second, second_target = start_worker()

    with Pipeline([first_target, second_target], DIGEST, cuts=[2]) as pipeline:
        pipeline.load()
        generated = pipeline.generate([1, 2, 3], max_new_tokens=4)

    # 3 -> 7, then 7 -> 1, 1 -> 5, 5 -> 9
    assert generated.tokens == [7, 1, 5, 9]
    assert generated.logits is None
    assert first.positions == second.positions == [0, 3, 4, 5]
    assert [(r.layer_begin, r.layer_end) for r in first.loaded] == [(0, 2)]
    assert [(r.layer_begin, r.layer_end) for r in second.loaded] == [(2, 0)]
    assert first.loaded[0].model_sha256 == DIGEST


def test_one_worker_holds_every_layer(start_worker):
    worker, target = start_worker()
    with Pipeline([target], DIGEST) as pipeline:
        pipeline.load()
        assert pipeline.generate([1, 2, 3], max_new_tokens=4).tokens == [7, 1, 5, 9]
    assert (worker.loaded[0].layer_begin, worker.loaded[0].layer_end) == (0, 0)


def test_logits_come_back_row_by_row(start_worker):
    _, first_target = start_worker()
    _, second_target = start_worker()
    with Pipeline([first_target, second_target], DIGEST, cuts=[2]) as pipeline:
        pipeline.load()
        generated = pipeline.generate([1, 2, 3], max_new_tokens=3, logits=True)

    # one row per prompt token, then one per decoded token that was fed back
    assert (generated.rows, generated.vocab) == (3 + 2, VOCAB)
    rows = array("f")
    rows.frombytes(generated.logits)
    argmax = [max(range(VOCAB), key=lambda j: rows[r * VOCAB + j]) for r in range(generated.rows)]
    assert argmax == [5, 6, 7, 1, 5]


def test_the_session_is_closed_on_every_worker(start_worker):
    first, first_target = start_worker()
    second, second_target = start_worker()
    with Pipeline([first_target, second_target], DIGEST, cuts=[2]) as pipeline:
        pipeline.load()
        pipeline.generate([1], max_new_tokens=2)
        pipeline.generate([1], max_new_tokens=2)
    assert len(first.closed) == len(second.closed) == 2
    assert len(set(first.closed)) == 2, "each generation has its own session"


def test_a_worker_error_reaches_the_caller_and_the_session_is_still_closed(start_worker):
    worker, target = start_worker(fail=True)
    with Pipeline([target], DIGEST) as pipeline:
        pipeline.load()
        with pytest.raises(grpc.RpcError) as error:
            pipeline.generate([1, 2], max_new_tokens=2)
    assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert len(worker.closed) == 1


def test_a_worker_that_stops_answering_times_out(start_worker):
    _, target = start_worker(hang=True)
    with Pipeline([target], DIGEST, step_timeout=0.3) as pipeline:
        pipeline.load()
        started = time.monotonic()
        with pytest.raises(PipelineError, match="did not answer"):
            pipeline.generate([1, 2], max_new_tokens=2)
    assert time.monotonic() - started < 10


def test_workers_that_do_not_form_a_pipeline_are_refused(start_worker):
    _, honest = start_worker()
    _, liar = start_worker(lie_about_shard=True)
    with Pipeline([honest, liar], DIGEST, cuts=[2]) as pipeline:
        with pytest.raises(PipelineError):
            pipeline.load()


def test_a_pipeline_must_be_loaded_before_it_generates(start_worker):
    _, target = start_worker()
    with Pipeline([target], DIGEST) as pipeline, pytest.raises(PipelineError, match="load"):
        pipeline.generate([1], max_new_tokens=1)


@pytest.mark.parametrize(
    ("targets", "cuts"),
    [
        ([], []),
        (["a:1", "b:2"], []),
        (["a:1"], [2]),
        (["a:1", "b:2", "c:3"], [3, 3]),
        (["a:1", "b:2", "c:3"], [5, 2]),
        (["a:1", "b:2"], [0]),
    ],
)
def test_bad_splits_are_refused(targets, cuts):
    with pytest.raises(ValueError):
        Pipeline(targets, DIGEST, cuts)


@pytest.mark.parametrize(("prompt", "count"), [([], 1), ([1], 0)])
def test_generate_needs_a_prompt_and_a_count(start_worker, prompt, count):
    _, target = start_worker()
    with Pipeline([target], DIGEST) as pipeline:
        pipeline.load()
        with pytest.raises(ValueError):
            pipeline.generate(prompt, count)


def test_cli_generate_prints_the_tokens_and_writes_the_logits(start_worker, tmp_path, capsys):
    _, first_target = start_worker()
    _, second_target = start_worker()
    logits_path = tmp_path / "logits.bin"

    code = cli.main(
        [
            "generate",
            "--model", DIGEST,
            "--stage", first_target,
            "--stage", second_target,
            "--cuts", "2",
            "--tokens", "1,2,3",
            "--max-new-tokens", "4",
            "--logits-out", str(logits_path),
        ]
    )  # fmt: skip

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "7,1,5,9"
    assert "layers [0, 2)" in captured.err and "layers [2, 4)" in captured.err
    raw = logits_path.read_bytes()
    assert struct.unpack("<ii", raw[:8]) == (3 + 3, VOCAB)
    assert len(raw) == 8 + 6 * VOCAB * 4


def test_cli_generate_reports_a_worker_error(start_worker, capsys):
    _, target = start_worker(fail=True)
    code = cli.main(["generate", "--model", DIGEST, "--stage", target, "--tokens", "1"])
    assert code == 1
    assert "FAILED_PRECONDITION: wrong position" in capsys.readouterr().err


def test_cli_generate_rejects_bad_numbers(capsys):
    code = cli.main(["generate", "--model", DIGEST, "--stage", "x:1", "--tokens", "1,,2"])
    assert code == 2
    assert "comma-separated integers" in capsys.readouterr().err
