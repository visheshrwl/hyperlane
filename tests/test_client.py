import socket
from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest

from shardline import cli
from shardline.client import first_reachable
from shardline.discovery import WorkerAdvert


def start_server():
    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, port


@pytest.fixture
def live_port():
    server, port = start_server()
    yield port
    server.stop(grace=None)


@pytest.fixture
def closed_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_first_reachable_skips_addresses_that_do_not_answer(live_port, closed_port):
    targets = [f"127.0.0.1:{closed_port}", f"127.0.0.1:{live_port}"]
    assert first_reachable(targets, timeout=2.0) == f"127.0.0.1:{live_port}"


def test_first_reachable_keeps_the_order_it_was_given():
    first, first_port = start_server()
    second, second_port = start_server()
    try:
        targets = [f"127.0.0.1:{first_port}", f"127.0.0.1:{second_port}"]
        assert first_reachable(targets, timeout=2.0) == targets[0]
        assert first_reachable(targets[::-1], timeout=2.0) == targets[1]
    finally:
        first.stop(grace=None)
        second.stop(grace=None)


def test_first_reachable_is_none_when_nothing_answers(closed_port):
    assert first_reachable([f"127.0.0.1:{closed_port}"], timeout=0.5) is None


def test_first_reachable_of_no_targets_is_none():
    assert first_reachable([]) is None


class FakeManager:
    def __init__(self, workers):
        self._workers = workers

    def start(self):
        pass

    def stop(self):
        pass

    def workers(self):
        return self._workers


def advert(addresses, port):
    return WorkerAdvert(
        name="w._shardline._tcp.local.",
        worker_id="w",
        addresses=addresses,
        port=port,
        protocol="shardline.v1",
        device_kind="cpu",
        device_name="Test CPU",
        total_memory_bytes=2**30,
        usable_memory_bytes=2**30,
    )


def test_discover_prints_the_address_that_answers(monkeypatch, capsys, live_port):
    # The worker listens on IPv4 only, so the IPv6 address that is listed first does not answer.
    worker = advert(("::1", "127.0.0.1"), live_port)
    monkeypatch.setattr("shardline.discovery.DiscoveryManager", lambda: FakeManager([worker]))

    assert cli.main(["discover", "--wait", "0"]) == 0

    out = capsys.readouterr().out
    assert f"127.0.0.1:{live_port}" in out
    assert "[::1]" not in out
    assert "no announced address" not in out


def test_discover_says_when_no_address_answers(monkeypatch, capsys, closed_port):
    worker = advert(("127.0.0.1",), closed_port)
    monkeypatch.setattr("shardline.discovery.DiscoveryManager", lambda: FakeManager([worker]))

    assert cli.main(["discover", "--wait", "0"]) == 0

    out = capsys.readouterr().out
    assert f"127.0.0.1:{closed_port}" in out
    assert "no announced address answers from here" in out
