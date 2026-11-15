"""Finds shardline workers on the local network over mDNS."""

from __future__ import annotations

import itertools
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from zeroconf import ServiceBrowser, ServiceInfo, ServiceStateChange, Zeroconf

SERVICE_TYPE = "_shardline._tcp.local."
TXT_VERSION = "1"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerAdvert:
    """What a worker announces about itself. GetStats is authoritative; these are hints."""

    name: str
    worker_id: str
    addresses: tuple[str, ...]
    port: int
    protocol: str
    device_kind: str
    device_name: str
    total_memory_bytes: int
    usable_memory_bytes: int

    @property
    def targets(self) -> tuple[str, ...]:
        """Every announced address as a gRPC target, in the order zeroconf reported them."""
        return tuple(
            f"[{host}]:{self.port}" if ":" in host else f"{host}:{self.port}"
            for host in self.addresses
        )

    @property
    def target(self) -> str:
        """The first announced address. A host with several interfaces announces all of
        them and this one may not be reachable from here; see client.first_reachable."""
        return self.targets[0]


def _to_int(value: str | None) -> int:
    try:
        return int(value or "")
    except ValueError:
        return 0


def parse_service_info(info: ServiceInfo) -> WorkerAdvert | None:
    properties = {
        key.decode(errors="replace"): (value.decode(errors="replace") if value is not None else "")
        for key, value in (info.properties or {}).items()
    }
    if properties.get("txtvers") != TXT_VERSION:
        return None
    addresses = tuple(info.parsed_scoped_addresses())
    if not addresses or not info.port:
        return None
    return WorkerAdvert(
        name=info.name,
        worker_id=properties.get("id") or info.name,
        addresses=addresses,
        port=info.port,
        protocol=properties.get("proto", ""),
        device_kind=properties.get("kind", ""),
        device_name=properties.get("device", ""),
        total_memory_bytes=_to_int(properties.get("mem_total")),
        usable_memory_bytes=_to_int(properties.get("mem_usable")),
    )


class DiscoveryManager:
    """Browses for workers in the background; no sockets are opened until start()."""

    def __init__(self, zeroconf_factory=Zeroconf):
        self._zeroconf_factory = zeroconf_factory
        self._zeroconf = None
        self._browser = None
        self._executor = None
        self._lock = threading.Lock()
        self._workers: dict[str, WorkerAdvert] = {}
        self._events = itertools.count(1)
        self._latest_event: dict[str, int] = {}

    def start(self) -> None:
        self._zeroconf = self._zeroconf_factory()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="shardline-mdns")
        self._browser = ServiceBrowser(self._zeroconf, SERVICE_TYPE, handlers=[self._on_change])

    def stop(self) -> None:
        if self._browser is not None:
            self._browser.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        if self._zeroconf is not None:
            self._zeroconf.close()
        self._browser = self._executor = self._zeroconf = None

    def workers(self) -> list[WorkerAdvert]:
        with self._lock:
            return sorted(self._workers.values(), key=lambda worker: worker.name)

    def _on_change(self, zeroconf, service_type, name, state_change) -> None:
        # The newest event for a name wins. A resolve that finishes after a later
        # event for the same name is stale and must not store its result.
        with self._lock:
            if state_change is ServiceStateChange.Removed:
                self._latest_event.pop(name, None)
                self._workers.pop(name, None)
                return
            event = self._latest_event[name] = next(self._events)
        # Resolving blocks, so it must not run on the browser thread.
        future = self._executor.submit(self._resolve, zeroconf, service_type, name, event)
        future.add_done_callback(lambda done: self._log_failure(name, done))

    def _resolve(self, zeroconf, service_type, name, event) -> None:
        info = zeroconf.get_service_info(service_type, name, timeout=3000)
        advert = parse_service_info(info) if info is not None else None
        if advert is None:
            return
        with self._lock:
            if self._latest_event.get(name) == event:
                self._workers[name] = advert

    @staticmethod
    def _log_failure(name: str, future: Future) -> None:
        if not future.cancelled() and future.exception() is not None:
            logger.warning("could not resolve %s", name, exc_info=future.exception())
