#!/usr/bin/env python3
"""
Unit tests for Hyperlane components.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "hyperlane_client"))

import unittest
from unittest.mock import Mock, patch
from hyperlane_client.discovery import DiscoveryManager
from hyperlane_client.orchestrator import KnapsackPartitioner


class TestKnapsackPartitioner(unittest.TestCase):
    """Test the knapsack partitioning algorithm."""

    def test_basic_partition(self):
        """Test basic layer partitioning."""
        layer_sizes = [100, 200, 150, 300, 250, 180]  # 1180 total
        worker_vram = {
            "worker_0": 500,
            "worker_1": 500,
            "worker_2": 500,
        }

        partitioner = KnapsackPartitioner()
        assignment = partitioner.partition(layer_sizes, worker_vram)

        # Check all layers are assigned
        all_layers = []
        for indices in assignment.values():
            all_layers.extend(indices)
        self.assertEqual(sorted(all_layers), list(range(len(layer_sizes))))

        # Check no worker exceeds capacity
        for worker_name, indices in assignment.items():
            total = sum(layer_sizes[i] for i in indices)
            self.assertLessEqual(total, worker_vram[worker_name])

    def test_single_worker(self):
        """Test with a single worker."""
        layer_sizes = [100, 200, 150]
        worker_vram = {"worker_0": 500}

        partitioner = KnapsackPartitioner()
        assignment = partitioner.partition(layer_sizes, worker_vram)

        self.assertEqual(len(assignment["worker_0"]), 3)
        self.assertEqual(assignment["worker_0"], [0, 1, 2])

    def test_insufficient_vram(self):
        """Test error when insufficient VRAM."""
        layer_sizes = [100, 200, 150]
        worker_vram = {"worker_0": 100}

        partitioner = KnapsackPartitioner()
        with self.assertRaises(ValueError):
            partitioner.partition(layer_sizes, worker_vram)


class TestDiscoveryManager(unittest.TestCase):
    """Test service discovery."""

    def test_initialization(self):
        """Test DiscoveryManager init."""
        manager = DiscoveryManager()
        self.assertIsNotNone(manager)
        self.assertEqual(len(manager.discovered_workers), 0)

    def test_add_worker(self):
        """Test manual worker addition."""
        manager = DiscoveryManager()
        worker = {
            "name": "test_worker",
            "address": "192.168.1.100",
            "port": 50051,
            "gpu_name": "RTX3090",
            "total_memory": 24 * 1024 * 1024 * 1024,
            "free_memory": 20 * 1024 * 1024 * 1024,
        }

        manager.discovered_workers["test_worker"] = worker

        workers = manager.get_workers()
        self.assertEqual(len(workers), 1)
        self.assertEqual(workers[0]["name"], "test_worker")

    def test_get_available_vram(self):
        """Test VRAM query."""
        manager = DiscoveryManager()
        manager.discovered_workers["worker_0"] = {
            "name": "worker_0",
            "address": "192.168.1.100",
            "port": 50051,
            "free_memory": 20,
        }
        manager.discovered_workers["worker_1"] = {
            "name": "worker_1",
            "address": "192.168.1.101",
            "port": 50052,
            "free_memory": 24,
        }

        vram = manager.get_available_vram()
        self.assertEqual(vram["worker_0"], 20)
        self.assertEqual(vram["worker_1"], 24)


if __name__ == "__main__":
    unittest.main()
