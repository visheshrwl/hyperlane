import logging
import socket
from concurrent.futures import Future

from zeroconf import ServiceInfo, ServiceStateChange

from shardline.discovery import SERVICE_TYPE, DiscoveryManager, parse_service_info

# Mirrors build_txt_record() in worker/src/discovery.cc.
WORKER_TXT = {
    "txtvers": "1",
    "proto": "shardline.v1",
    "id": "mac-mini-1",
    "devices": "2",
    "kind": "metal",
    "device": "Apple M5",
    "mem_total": "17179869184",
    "mem_usable": "12713115648",
}


LAN_ADDRESS = socket.inet_aton("192.168.1.20")


def service_info(properties, addresses=(LAN_ADDRESS,), port=50051):
    return ServiceInfo(
        SERVICE_TYPE,
        f"mac-mini-1.{SERVICE_TYPE}",
        port=port,
        properties=properties,
        addresses=list(addresses),
    )


def test_parses_one_txt_key_per_field():
    advert = parse_service_info(service_info(WORKER_TXT))

    assert advert is not None
    assert advert.worker_id == "mac-mini-1"
    assert advert.protocol == "shardline.v1"
    assert advert.device_kind == "metal"
    assert advert.device_name == "Apple M5"
    assert advert.total_memory_bytes == 17179869184
    assert advert.usable_memory_bytes == 12713115648
    assert advert.target == "192.168.1.20:50051"


def test_ipv6_target_is_bracketed():
    address = socket.inet_pton(socket.AF_INET6, "fd00::20")
    advert = parse_service_info(service_info(WORKER_TXT, addresses=(address,)))
    assert advert.target == "[fd00::20]:50051"


def test_rejects_unknown_txt_version():
    assert parse_service_info(service_info({**WORKER_TXT, "txtvers": "2"})) is None


def test_malformed_memory_reads_as_zero():
    advert = parse_service_info(service_info({**WORKER_TXT, "mem_usable": "lots"}))
    assert advert.usable_memory_bytes == 0


def test_no_sockets_until_started():
    created = []
    manager = DiscoveryManager(zeroconf_factory=lambda: created.append(1))
    assert created == []
    assert manager.workers() == []


def test_removed_service_is_forgotten():
    manager = DiscoveryManager()
    advert = parse_service_info(service_info(WORKER_TXT))
    manager._workers[advert.name] = advert

    manager._on_change(None, SERVICE_TYPE, advert.name, ServiceStateChange.Removed)

    assert manager.workers() == []


def test_every_announced_address_is_a_target():
    addresses = (LAN_ADDRESS, socket.inet_aton("100.64.0.3"))
    advert = parse_service_info(service_info(WORKER_TXT, addresses=addresses))

    assert set(advert.targets) == {"192.168.1.20:50051", "100.64.0.3:50051"}
    assert advert.target == advert.targets[0]


def test_undecodable_txt_bytes_do_not_drop_the_worker():
    advert = parse_service_info(service_info({**WORKER_TXT, "device": b"Apple \xff\xfe"}))

    assert advert is not None
    assert advert.device_name.startswith("Apple ")
    assert advert.worker_id == "mac-mini-1"


NAME = f"mac-mini-1.{SERVICE_TYPE}"


class InlineExecutor:
    """Runs submitted work at once, in the calling thread."""

    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as error:
            future.set_exception(error)
        return future


class QueuedExecutor:
    """Holds submitted work until the test decides which to run, and when."""

    def __init__(self):
        self.pending = []

    def submit(self, fn, *args):
        future = Future()
        self.pending.append((future, fn, args))
        return future

    def run(self, index):
        future, fn, args = self.pending[index]
        try:
            future.set_result(fn(*args))
        except Exception as error:
            future.set_exception(error)


class FakeZeroconf:
    """Answers each lookup with the next item; an exception is raised instead."""

    def __init__(self, *answers, on_lookup=None):
        self._answers = list(answers)
        self._on_lookup = on_lookup

    def get_service_info(self, service_type, name, timeout):
        if self._on_lookup:
            self._on_lookup()
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_resolved_service_is_listed():
    manager = DiscoveryManager()
    manager._executor = InlineExecutor()
    zeroconf = FakeZeroconf(service_info(WORKER_TXT))

    manager._on_change(zeroconf, SERVICE_TYPE, NAME, ServiceStateChange.Added)

    assert [worker.worker_id for worker in manager.workers()] == ["mac-mini-1"]


def test_removal_during_a_pending_resolve_is_not_undone():
    manager = DiscoveryManager()
    manager._executor = InlineExecutor()
    zeroconf = FakeZeroconf(
        service_info(WORKER_TXT),
        on_lookup=lambda: manager._on_change(
            zeroconf, SERVICE_TYPE, NAME, ServiceStateChange.Removed
        ),
    )

    manager._on_change(zeroconf, SERVICE_TYPE, NAME, ServiceStateChange.Added)

    assert manager.workers() == []


def test_older_resolve_finishing_late_does_not_overwrite_a_newer_one():
    manager = DiscoveryManager()
    manager._executor = executor = QueuedExecutor()
    older = service_info({**WORKER_TXT, "mem_usable": "1"})
    newer = service_info({**WORKER_TXT, "mem_usable": "2"})
    zeroconf = FakeZeroconf(newer, older)  # answers follow the order the lookups run in

    manager._on_change(zeroconf, SERVICE_TYPE, NAME, ServiceStateChange.Added)
    manager._on_change(zeroconf, SERVICE_TYPE, NAME, ServiceStateChange.Updated)
    executor.run(1)
    executor.run(0)

    assert [worker.usable_memory_bytes for worker in manager.workers()] == [2]


def test_a_service_that_returns_after_removal_is_listed_again():
    manager = DiscoveryManager()
    manager._executor = InlineExecutor()
    zeroconf = FakeZeroconf(service_info(WORKER_TXT), service_info(WORKER_TXT))

    manager._on_change(zeroconf, SERVICE_TYPE, NAME, ServiceStateChange.Added)
    manager._on_change(zeroconf, SERVICE_TYPE, NAME, ServiceStateChange.Removed)
    manager._on_change(zeroconf, SERVICE_TYPE, NAME, ServiceStateChange.Added)

    assert [worker.worker_id for worker in manager.workers()] == ["mac-mini-1"]


def test_resolve_failure_is_logged(caplog):
    manager = DiscoveryManager()
    manager._executor = InlineExecutor()
    zeroconf = FakeZeroconf(RuntimeError("boom"))

    with caplog.at_level(logging.WARNING, logger="shardline.discovery"):
        manager._on_change(zeroconf, SERVICE_TYPE, NAME, ServiceStateChange.Added)

    assert NAME in caplog.text
    assert "boom" in caplog.text
    assert manager.workers() == []
