"""The shardline command line."""

from __future__ import annotations

import argparse
import sys
import time

GIB = 1024**3


def _stats(target: str) -> int:
    import grpc

    from shardline.client import WorkerClient
    from shardline.v1 import worker_pb2

    with WorkerClient(target) as client:
        try:
            stats = client.get_stats()
        except grpc.RpcError as error:
            print(f"{target}: {error.code().name}: {error.details()}", file=sys.stderr)
            return 1

    print(
        f"{stats.worker_id} on {stats.hostname} ({stats.protocol}, worker {stats.worker_version})"
    )
    for device in stats.devices:
        kind = worker_pb2.DeviceKind.Name(device.kind).removeprefix("DEVICE_KIND_").lower()
        print(
            f"  {kind:<6} {device.name:<32} usable {device.usable_memory_bytes / GIB:6.1f} GiB"
            f" of {device.total_memory_bytes / GIB:6.1f} GiB"
        )
    return 0


def _discover(wait: float) -> int:
    from shardline.discovery import DiscoveryManager

    manager = DiscoveryManager()
    manager.start()
    try:
        time.sleep(wait)
        workers = manager.workers()
    finally:
        manager.stop()

    if not workers:
        print(f"no workers found in {wait:g} s", file=sys.stderr)
        return 1
    for worker in workers:
        print(
            f"{worker.target:<24} {worker.worker_id:<32} {worker.device_kind:<6} "
            f"{worker.device_name} ({worker.usable_memory_bytes / GIB:.1f} GiB usable)"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shardline")
    commands = parser.add_subparsers(dest="command", required=True)

    stats = commands.add_parser("stats", help="show a worker's devices and memory")
    stats.add_argument("target", help="worker address, HOST:PORT")

    discover = commands.add_parser("discover", help="list workers advertised over mDNS")
    discover.add_argument("--wait", type=float, default=3.0, help="seconds to listen")

    args = parser.parse_args(argv)
    if args.command == "stats":
        return _stats(args.target)
    return _discover(args.wait)


if __name__ == "__main__":
    sys.exit(main())
