"""
Service discovery via zeroconf (mDNS).
"""

from zeroconf import ServiceBrowser, Zeroconf
import socket
from typing import Dict, Optional, List


class DiscoveryManager:
    """Discovers Hyperlane workers on the LAN via _hyperlane._tcp service."""

    def __init__(self):
        self.zeroconf = Zeroconf()
        self.discovered_workers: Dict[str, Dict] = {}
        self.browser = None

    def start_discovery(self):
        """Start listening for _hyperlane._tcp services."""
        from zeroconf import ServiceStateChange

        def on_service_change(zeroconf, service_type, name, state_change):
            if state_change == ServiceStateChange.Added:
                self._on_worker_discovered(zeroconf, service_type, name)
            elif state_change == ServiceStateChange.Removed:
                self._on_worker_removed(name)

        self.browser = ServiceBrowser(
            self.zeroconf, "_hyperlane._tcp.local.", handlers=[on_service_change]
        )

    def stop_discovery(self):
        """Stop the discovery browser."""
        if self.browser:
            self.browser.cancel()
        self.zeroconf.close()

    def _on_worker_discovered(self, zeroconf, service_type, name):
        """Callback when a worker is discovered."""
        from zeroconf import Zeroconf as ZC

        info = zeroconf.get_service_info(service_type, name)
        if info:
            address = socket.inet_ntoa(info.addresses[0])
            port = info.port
            properties = info.properties or {}

            worker_info = {
                "name": name,
                "address": address,
                "port": port,
                "gpu_name": properties.get(b"gpu_name", b"Unknown").decode(),
                "total_memory": int(
                    properties.get(b"total_memory", b"0").decode()
                ),
                "free_memory": int(properties.get(b"free_memory", b"0").decode()),
            }

            self.discovered_workers[name] = worker_info
            print(f"[Discovery] Worker discovered: {name} at {address}:{port}")

    def _on_worker_removed(self, name):
        """Callback when a worker leaves."""
        if name in self.discovered_workers:
            del self.discovered_workers[name]
            print(f"[Discovery] Worker removed: {name}")

    def get_workers(self) -> List[Dict]:
        """Return list of discovered workers."""
        return list(self.discovered_workers.values())

    def get_available_vram(self) -> Dict[str, int]:
        """Return VRAM availability per worker."""
        return {
            worker["name"]: worker["free_memory"]
            for worker in self.discovered_workers.values()
        }
