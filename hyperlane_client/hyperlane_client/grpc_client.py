"""
gRPC client for orchestrating Hyperlane workers.
"""

import grpc
from typing import List, Dict, Optional
import sys
from pathlib import Path

# Add parent dir to path for proto imports
sys.path.insert(0, str(Path(__file__).parent))

# Note: These imports will work after generate_grpc.py is run
# For now, we provide a stub that can be filled in after proto compilation
# from . import service_pb2, service_pb2_grpc


class WorkerClient:
    """gRPC client for a single worker."""

    def __init__(self, address: str, port: int):
        self.address = address
        self.port = port
        self.channel = None
        self.stub = None

    def connect(self):
        """Establish gRPC channel."""
        target = f"{self.address}:{self.port}"
        self.channel = grpc.aio.secure_channel(
            target, grpc.ssl_channel_credentials()
        ) if False else grpc.aio.insecure_channel(target)
        # TODO: Import and initialize stub after proto generation
        # self.stub = service_pb2_grpc.WorkerStub(self.channel)

    async def disconnect(self):
        """Close gRPC channel."""
        if self.channel:
            await self.channel.close()

    async def get_stats(self):
        """Query worker GPU stats."""
        # TODO: Call self.stub.GetStats()
        return None

    async def load_shard(self, shard_name: str, model_path: str,
                        next_worker_address: str = ""):
        """Load an ONNX shard on the worker."""
        # TODO: Call self.stub.LoadShard()
        pass

    async def execute_pipeline(self, run_id: str):
        """Trigger execution on the worker."""
        # TODO: Call self.stub.ExecutePipeline()
        pass

    async def set_next_worker(self, address: str):
        """Configure the next worker in the pipeline."""
        # TODO: Call self.stub.SetNextWorker()
        pass


class WorkerPool:
    """Manages connections to multiple workers."""

    def __init__(self, workers: List[Dict]):
        self.workers = {w["name"]: WorkerClient(w["address"], w["port"]) 
                       for w in workers}

    async def connect_all(self):
        """Connect to all workers."""
        for client in self.workers.values():
            client.connect()

    async def disconnect_all(self):
        """Disconnect from all workers."""
        for client in self.workers.values():
            await client.disconnect()

    async def load_shards(self, shard_assignment: Dict[str, List[int]],
                         onnx_files: Dict[str, str]):
        """Load shards on their assigned workers."""
        # TODO: Iterate through shard_assignment and call load_shard on each worker
        pass
