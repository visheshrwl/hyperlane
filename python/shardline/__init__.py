"""
hyperlane_client: Distributed GPU inference orchestration library.
"""

__version__ = "0.1.0"

from .discovery import DiscoveryManager
from .orchestrator import AutoDistributedModel

__all__ = ["DiscoveryManager", "AutoDistributedModel"]
