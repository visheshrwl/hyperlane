"""shardline: pipeline-parallel LLM serving across mixed NVIDIA and Apple Silicon machines."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("shardline")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
