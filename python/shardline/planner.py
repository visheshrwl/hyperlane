"""
Model sharding and orchestration.
"""

from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import os


class KnapsackPartitioner:
    """Partitions model layers into shards via a knapsack algorithm."""

    @staticmethod
    def partition(
        layer_sizes: List[int], worker_vram: Dict[str, int]
    ) -> Dict[str, List[int]]:
        """
        Partition layers across workers using a greedy bin-packing approach.

        Args:
            layer_sizes: List of layer sizes (bytes) in order.
            worker_vram: Dict of worker_name -> available_vram.

        Returns:
            Dict mapping worker_name to list of layer indices assigned to it.
        """
        workers = list(worker_vram.keys())
        worker_capacities = {w: worker_vram[w] for w in workers}
        shard_assignment = {w: [] for w in workers}
        worker_loads = {w: 0 for w in workers}

        # Greedy: assign each layer to the worker with most remaining capacity
        for layer_idx, layer_size in enumerate(layer_sizes):
            # Find worker with most free capacity
            best_worker = max(
                workers,
                key=lambda w: worker_vram[w] - worker_loads[w],
            )

            remaining = worker_vram[best_worker] - worker_loads[best_worker]
            if remaining < layer_size:
                raise ValueError(
                    f"Layer {layer_idx} (size {layer_size}) does not fit in any worker"
                )

            shard_assignment[best_worker].append(layer_idx)
            worker_loads[best_worker] += layer_size

        return shard_assignment


class AutoDistributedModel:
    """Auto-shards a PyTorch model across discovered workers."""

    def __init__(self, discovery_manager):
        self.discovery_manager = discovery_manager
        self.model = None
        self.tokenizer = None
        self.shard_assignment = None
        self.worker_clients = {}
        self.onnx_shards = {}  # shard_name -> onnx_path

    @classmethod
    def from_pretrained(cls, model_name: str, discovery_manager):
        """
        Load a model from HuggingFace Hub and auto-shard across discovered workers.

        Args:
            model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b").
            discovery_manager: DiscoveryManager instance with discovered workers.

        Returns:
            AutoDistributedModel instance, ready for inference.
        """
        instance = cls(discovery_manager)

        # Ensure workers are discovered
        if not discovery_manager.discovered_workers:
            raise RuntimeError(
                "No workers discovered. Start discovery first."
            )

        # Load model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[Sharding] Loading model: {model_name}")
        instance.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="cpu"
        )
        instance.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Extract transformer layers for sharding
        layers = instance._extract_transformer_layers()
        layer_sizes = instance._estimate_layer_sizes(layers)

        # Get worker VRAM
        worker_vram = discovery_manager.get_available_vram()
        if not worker_vram:
            raise RuntimeError("No workers available for sharding.")

        # Partition layers
        print(f"[Sharding] Partitioning {len(layers)} layers across {len(worker_vram)} workers")
        partitioner = KnapsackPartitioner()
        instance.shard_assignment = partitioner.partition(
            layer_sizes, worker_vram
        )

        print(f"[Sharding] Assignment: {instance.shard_assignment}")

        # Convert layers to ONNX shards
        instance._export_shards(layers)

        # Deploy shards to workers
        instance._deploy_shards()

        return instance

    def _extract_transformer_layers(self) -> List[nn.Module]:
        """Extract transformer layers from model."""
        layers = []

        # Handle different model architectures
        if hasattr(self.model, "gpt_neox"):  # GPT-NeoX (Pythia, Stability)
            layers = list(self.model.gpt_neox.layers)
        elif hasattr(self.model, "transformer"):  # GPT-2/3
            layers = list(self.model.transformer.h)
        elif hasattr(self.model, "model"):
            if hasattr(self.model.model, "layers"):  # LLaMA
                layers = list(self.model.model.layers)
            elif hasattr(self.model.model, "h"):  # Mistral
                layers = list(self.model.model.h)

        if not layers:
            raise ValueError(f"Could not extract layers from {type(self.model)}")

        return layers

    def _estimate_layer_sizes(self, layers: List[nn.Module]) -> List[int]:
        """Estimate memory size of each layer in bytes."""
        sizes = []
        for layer in layers:
            size = 0
            for param in layer.parameters():
                size += param.numel() * param.element_size()
            sizes.append(size)
        return sizes

    def _export_shards(self, layers: List[nn.Module]):
        """Convert layer groups to ONNX format."""
        print("[Sharding] Exporting ONNX shards...")

        # Create temp directory for ONNX files
        temp_dir = tempfile.mkdtemp(prefix="hyperlane_shards_")
        print(f"[Sharding] Using temp dir: {temp_dir}")

        for worker_name, layer_indices in self.shard_assignment.items():
            if not layer_indices:
                continue

            shard_name = f"shard_{worker_name}"
            onnx_path = os.path.join(temp_dir, f"{shard_name}.onnx")

            # Create a wrapper module for these layers
            shard_module = nn.Sequential(
                *[layers[i] for i in layer_indices]
            )

            # Export to ONNX (simplified)
            try:
                print(f"[Sharding] Exporting {shard_name} ({len(layer_indices)} layers) to {onnx_path}")

                # Create dummy input (batch_size=1, seq_len=1, hidden_dim=4096)
                dummy_input = torch.randn(1, 1, 4096, dtype=torch.float16)

                torch.onnx.export(
                    shard_module,
                    (dummy_input,),
                    onnx_path,
                    opset_version=17,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch_size", 1: "seq_len"},
                        "output": {0: "batch_size", 1: "seq_len"},
                    },
                    do_constant_folding=False,
                )

                self.onnx_shards[shard_name] = onnx_path
                print(f"[Sharding] Exported {shard_name}")

            except Exception as e:
                print(f"[Sharding] Failed to export {shard_name}: {e}")

    def _deploy_shards(self):
        """Deploy ONNX shards to workers via gRPC."""
        from .grpc_client import WorkerPool

        print("[Deployment] Starting shard deployment...")

        # Create worker pool
        workers = self.discovery_manager.get_workers()
        worker_pool = WorkerPool(workers)

        try:
            import asyncio

            async def deploy():
                await worker_pool.connect_all()

                # Load each shard on its assigned worker
                for shard_name, onnx_path in self.onnx_shards.items():
                    # Find which worker this shard belongs to
                    for worker_name, layer_indices in self.shard_assignment.items():
                        if f"shard_{worker_name}" == shard_name:
                            # Find worker by name
                            for worker_info in workers:
                                if worker_info["name"] == worker_name:
                                    client = worker_pool.workers.get(worker_info["name"])
                                    if client:
                                        print(f"[Deployment] Loading {shard_name} on {worker_name}")
                                        # TODO: Actually call gRPC LoadShard
                                        # await client.load_shard(shard_name, onnx_path)
                                    break
                            break

                await worker_pool.disconnect_all()

            # Run deployment
            asyncio.run(deploy())

        except ImportError:
            print("[Deployment] asyncio not available, skipping async deployment")

        print("[Deployment] Shard deployment complete")

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        """
        Run inference on the distributed model.

        Uses a pybind11 C++ extension for initial tensor transmission.
        """
        print(f"[Generate] Prompt: {prompt}")

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        print(f"[Generate] Input shape: {input_ids.shape}")

        # TODO: Send initial input to first worker via TensorSocket
        # TODO: Pipeline stages execute in parallel
        # TODO: Collect output from last worker

        return "Generated text placeholder"
