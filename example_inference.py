#!/usr/bin/env python3
"""
Example: Discover workers and load a distributed model.
"""

import asyncio
import sys
from pathlib import Path

# Add client to path
sys.path.insert(0, str(Path(__file__).parent / "hyperlane_client"))

from hyperlane_client import DiscoveryManager, AutoDistributedModel


async def main():
    print("=" * 60)
    print("Hyperlane Distributed Inference Example")
    print("=" * 60)
    print()

    # Initialize discovery
    print("[1] Initializing service discovery...")
    discovery = DiscoveryManager()
    discovery.start_discovery()

    # Wait for workers to be discovered
    print("[2] Waiting for workers (5 seconds)...")
    for i in range(5):
        print(f"  {5 - i} seconds...", flush=True)
        await asyncio.sleep(1)

    workers = discovery.get_workers()
    print(f"\n[✓] Found {len(workers)} worker(s)")

    if not workers:
        print("\n[!] No workers discovered. Make sure to start workers with:")
        print("    ./hyperlane_worker/build/hyperlane_worker [port]")
        discovery.stop_discovery()
        return

    for w in workers:
        print(f"    - {w['name']} at {w['address']}:{w['port']}")
        print(f"      GPU: {w.get('gpu_name', 'N/A')}")
        print(f"      VRAM: {w.get('free_memory', 0) / 1e9:.1f} GB free")

    # Load model (this is where sharding happens)
    print("\n[3] Loading model with auto-sharding...")
    print("    (This may take a few minutes for large models)")

    try:
        model = AutoDistributedModel.from_pretrained(
            "meta-llama/Llama-2-7b-hf",  # ~7B parameters
            discovery,
        )
        print("[✓] Model loaded and sharded")

        # Run inference
        print("\n[4] Running inference...")
        prompt = "What is machine learning?"
        result = model.generate(prompt, max_tokens=128)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {result}")

    except Exception as e:
        print(f"[!] Error: {e}")

    # Cleanup
    print("\n[5] Cleanup...")
    discovery.stop_discovery()
    print("[✓] Done")


if __name__ == "__main__":
    asyncio.run(main())
