#!/usr/bin/env python3
"""
Generate gRPC Python stubs from service.proto.
"""

import subprocess
import sys
from pathlib import Path

def main():
    proto_dir = Path(__file__).parent.parent / "proto"
    proto_file = proto_dir / "service.proto"
    
    if not proto_file.exists():
        print(f"Error: {proto_file} not found", file=sys.stderr)
        return 1
    
    output_dir = Path(__file__).parent / "hyperlane_client"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python", "-m", "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        str(proto_file),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
