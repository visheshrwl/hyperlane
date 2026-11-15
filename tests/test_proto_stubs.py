from pathlib import Path

import pytest

from shardline.v1 import worker_pb2

PROTO = Path(__file__).parent.parent / "proto" / "shardline" / "v1" / "worker.proto"


@pytest.mark.skipif(not PROTO.exists(), reason="not a source checkout")
def test_generated_stubs_are_not_older_than_the_proto():
    # The stubs are generated when the package is installed, not when the proto changes.
    assert Path(worker_pb2.__file__).stat().st_mtime >= PROTO.stat().st_mtime, (
        "worker_pb2.py is older than worker.proto; regenerate it with "
        "`uv sync --reinstall-package shardline`"
    )
