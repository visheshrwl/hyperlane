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
    from shardline.client import first_reachable
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
        target = first_reachable(worker.targets)
        note = "" if target else "  [no announced address answers from here]"
        print(
            f"{target or worker.target:<24} {worker.worker_id:<32} {worker.device_kind:<6} "
            f"{worker.device_name} ({worker.usable_memory_bytes / GIB:.1f} GiB usable){note}"
        )
    return 0


def _model_add(path: str, model_dir: str | None) -> int:
    from pathlib import Path

    from shardline.models import add_model, default_model_dir

    directory = Path(model_dir) if model_dir else default_model_dir()
    try:
        digest = add_model(Path(path), directory)
    except (OSError, ValueError) as error:
        print(f"{path}: {error}", file=sys.stderr)
        return 1
    print(digest)
    return 0


DEVICES = {
    "auto": "DEVICE_KIND_UNSPECIFIED",
    "cpu": "DEVICE_KIND_CPU",
    "metal": "DEVICE_KIND_METAL",
    "cuda": "DEVICE_KIND_CUDA",
}


def _generate(args: argparse.Namespace) -> int:
    import struct
    from pathlib import Path

    import grpc

    from shardline.pipeline import Pipeline, PipelineError
    from shardline.v1 import worker_pb2

    try:
        prompt = [int(token) for token in args.tokens.split(",")]
        cuts = [int(cut) for cut in args.cuts.split(",")] if args.cuts else []
    except ValueError:
        print("--tokens and --cuts are comma-separated integers", file=sys.stderr)
        return 2

    try:
        with Pipeline(
            args.stage,
            args.model,
            cuts,
            device=getattr(worker_pb2, DEVICES[args.device]),
            context_length=args.context,
            step_timeout=args.step_timeout,
        ) as pipeline:
            for target, loaded in zip(args.stage, pipeline.load(), strict=True):
                shard = loaded.shard
                print(
                    f"{target}: layers [{shard.layer_begin}, {shard.layer_end}) "
                    f"on {shard.backend}, {shard.weight_bytes / GIB:.2f} GiB of weights",
                    file=sys.stderr,
                )
            result = pipeline.generate(prompt, args.max_new_tokens, logits=bool(args.logits_out))
    except grpc.RpcError as error:
        print(f"{error.code().name}: {error.details()}", file=sys.stderr)
        return 1
    except (PipelineError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    if args.logits_out:
        # The layout the parity scripts read: int32 rows, int32 columns, float32 data.
        with Path(args.logits_out).open("wb") as out:
            out.write(struct.pack("<ii", result.rows, result.vocab))
            out.write(result.logits)
    print(",".join(str(token) for token in result.tokens))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shardline")
    commands = parser.add_subparsers(dest="command", required=True)

    stats = commands.add_parser("stats", help="show a worker's devices and memory")
    stats.add_argument("target", help="worker address, HOST:PORT")

    discover = commands.add_parser("discover", help="list workers advertised over mDNS")
    discover.add_argument("--wait", type=float, default=3.0, help="seconds to listen")

    model = commands.add_parser("model", help="manage the models a worker serves")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    add = model_commands.add_parser("add", help="copy a GGUF file in under its SHA-256")
    add.add_argument("path", help="the GGUF file")
    add.add_argument(
        "--model-dir", help="default: $SHARDLINE_MODEL_DIR or ~/.cache/shardline/models"
    )

    generate = commands.add_parser("generate", help="greedily generate tokens across workers")
    generate.add_argument("--model", required=True, help="the model's SHA-256, from `model add`")
    generate.add_argument(
        "--stage", action="append", required=True, help="a worker, HOST:PORT; repeat per stage"
    )
    generate.add_argument("--cuts", help="layers where each later stage starts, e.g. 11 or 8,16")
    generate.add_argument("--tokens", required=True, help="the prompt as comma-separated token ids")
    generate.add_argument("--max-new-tokens", type=int, default=32)
    generate.add_argument("--device", choices=sorted(DEVICES), default="auto")
    generate.add_argument("--context", type=int, default=0, help="KV cache tokens; 0 is 512")
    generate.add_argument("--step-timeout", type=float, default=120.0, help="seconds per hop")
    generate.add_argument("--logits-out", help="also write every logits row to this file")

    args = parser.parse_args(argv)
    if args.command == "stats":
        return _stats(args.target)
    if args.command == "model":
        return _model_add(args.path, args.model_dir)
    if args.command == "generate":
        return _generate(args)
    return _discover(args.wait)


if __name__ == "__main__":
    sys.exit(main())
