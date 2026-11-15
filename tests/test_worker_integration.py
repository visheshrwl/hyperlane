"""Runs the real worker binary. Set SHARDLINE_WORKER_BIN to its path."""

import os
import queue
import re
import signal
import subprocess
import threading

import pytest

from shardline.cli import main
from shardline.client import WorkerClient
from shardline.v1 import worker_pb2

WORKER_BIN = os.environ.get("SHARDLINE_WORKER_BIN")

pytestmark = pytest.mark.skipif(not WORKER_BIN, reason="SHARDLINE_WORKER_BIN is not set")


STARTUP_TIMEOUT = 20


def first_line(stream, timeout):
    """The next line from stream, or None if none arrives within timeout seconds."""
    lines = queue.Queue()
    threading.Thread(target=lambda: lines.put(stream.readline()), daemon=True).start()
    try:
        return lines.get(timeout=timeout)
    except queue.Empty:
        return None


@pytest.fixture
def worker():
    process = subprocess.Popen(
        [WORKER_BIN, "--listen", "127.0.0.1:0", "--worker-id", "integration"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = first_line(process.stdout, STARTUP_TIMEOUT)
        assert line is not None, f"the worker printed nothing in {STARTUP_TIMEOUT} s"
        match = re.search(r"listening on (\S+)", line)
        assert match, f"unexpected worker output: {line!r}"
        yield process, match.group(1)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_get_stats_over_grpc(worker):
    _, target = worker
    with WorkerClient(target) as client:
        stats = client.get_stats()

    assert stats.worker_id == "integration"
    assert stats.protocol == "shardline.v1"
    assert stats.devices[-1].kind == worker_pb2.DEVICE_KIND_CPU
    for device in stats.devices:
        assert 0 < device.usable_memory_bytes <= device.total_memory_bytes


def test_cli_stats(worker, capsys):
    _, target = worker
    assert main(["stats", target]) == 0
    assert "integration" in capsys.readouterr().out


def test_sigterm_exits_cleanly(worker):
    process, _ = worker
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=10) == 0


def test_cli_stats_reports_unreachable_worker(capsys):
    assert main(["stats", "127.0.0.1:1"]) == 1
    assert "UNAVAILABLE" in capsys.readouterr().err
