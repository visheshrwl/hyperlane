"""Generates the shardline.v1 protobuf and gRPC modules at build time."""

from importlib import resources
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class ProtoBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        from grpc_tools import protoc

        root = Path(self.root)
        proto_dir = root / "proto"
        out_dir = root / "python"
        protos = sorted(str(p.relative_to(proto_dir)) for p in proto_dir.rglob("*.proto"))
        well_known = resources.files("grpc_tools") / "_proto"
        args = [
            "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--proto_path={well_known}",
            f"--python_out={out_dir}",
            f"--pyi_out={out_dir}",
            f"--grpc_python_out={out_dir}",
            *protos,
        ]
        if protoc.main(args) != 0:
            raise RuntimeError(f"protoc failed for {protos}")
