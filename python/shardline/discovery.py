"""Finds shardline workers on the local network over mDNS."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from zeroconf import ServiceBrowser, ServiceInfo, ServiceStateChange, Zeroconf

SERVICE_TYPE = "_shardline._tcp.local."
TXT_VERSION = "1"


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
    def target(self) -> str:
        host = self.addresses[0]
        return f"[{host}]:{self.port}" if ":" in host else f"{host}:{self.port}"


def _to_int(value: str | None) -> int:
    try:
        return int(value or "")
    except ValueError:
        return 0


def parse_service_info(info: ServiceInfo) -> WorkerAdvert | None:
    properties = {
        key.decode(): (value.decode() if value is not None else "")
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
        if state_change is ServiceStateChange.Removed:
            with self._lock:
                self._workers.pop(name, None)
            return
        # Resolving blocks, so it must not run on the browser thread.
        self._executor.submit(self._resolve, zeroconf, service_type, name)

    def _resolve(self, zeroconf, service_type, name) -> None:
        info = zeroconf.get_service_info(service_type, name, timeout=3000)
        advert = parse_service_info(info) if info is not None else None
        if advert is not None:
            with self._lock:
                self._workers[name] = advert
