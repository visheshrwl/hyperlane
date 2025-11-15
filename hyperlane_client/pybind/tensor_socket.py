"""
pybind11 module wrapper for high-performance tensor transmission.

This module exposes the C++ extension (hyperlane_tensor_socket) which sends tensors
directly from NumPy/PyTorch memory to a C++ TCP socket, bypassing Python's GIL
and enabling DMA-friendly pinned memory transfers.
"""

try:
    from hyperlane_tensor_socket import TensorSocket
except ImportError:
    # Fallback: Pure Python placeholder if C++ extension not built
    class TensorSocket:
        """Fallback Python implementation."""

        def __init__(self, address: str, port: int):
            self.address = address
            self.port = port
            self._socket = None

        def connect(self):
            """Establish connection (placeholder)."""
            pass

        def disconnect(self):
            """Close connection (placeholder)."""
            pass

        def async_send(self, tensor, name: str) -> int:
            """Send tensor asynchronously (placeholder)."""
            return -1

        def is_complete(self, handle: int) -> bool:
            """Check completion status (placeholder)."""
            return True

        def wait_complete(self, handle: int):
            """Wait for completion (placeholder)."""
            pass


__all__ = ["TensorSocket"]
