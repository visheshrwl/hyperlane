"""End-to-end check of the vertical slice, with real worker processes.

    uv run --extra oracle python scripts/e2e_smoke.py --backend cpu
    uv run --extra oracle python scripts/e2e_smoke.py --backend metal --model tinyllama

It starts workers on loopback, puts a model in their model directory, and generates
tokens through one worker, two and three. The answers must match:

- every split bit for bit with the single worker on the same backend, and
- Hugging Face's fp32 model on the CPU, with greedy tokens equal, top-1 agreement of at
  least 0.99 and a mean KL under 1e-3 per row (the tolerances the architecture doc
  proposes across hardware), plus max |dlogit| of 1e-3 when both sides run fp32 on the CPU.

--model tiny (the default) builds a random 4-layer Llama, needs no download and runs on
CI. --model tinyllama uses the files the Phase 1 spike left in .cache.
"""

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compare import compare, read_logits
from reference import greedy, load_model
from shardline.models import add_model
from shardline.pipeline import Pipeline
from shardline.v1 import worker_pb2

ROOT = Path(__file__).resolve().parent.parent
TINY_PROMPT = [5, 17, 3, 42, 8, 11, 2, 60]

MIN_TOP1 = 0.99
MAX_MEAN_KL = 1e-3
MAX_FP32_CPU_LOGIT_DIFF = 1e-3

DEVICES = {"cpu": worker_pb2.DEVICE_KIND_CPU, "metal": worker_pb2.DEVICE_KIND_METAL}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


class Worker:
    """A shardline-worker process on loopback."""

    def __init__(self, binary: Path, model_dir: Path, name: str, log_dir: Path, threads: int):
        self.log = (log_dir / f"{name}.err").open("w")
        self.process = subprocess.Popen(
            [
                str(binary),
                "--listen", "127.0.0.1:0",
                "--worker-id", name,
                "--model-dir", str(model_dir),
                "--threads", str(threads),
            ],
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
        )  # fmt: skip
        line = self._first_line(30)
        match = re.search(r"listening on (\S+)", line or "")
        if not match:
            self.stop()
            raise RuntimeError(f"{name} did not start: {line!r}; see {self.log.name}")
        self.target = match.group(1)

    def _first_line(self, timeout: float) -> str | None:
        lines: queue.Queue[str] = queue.Queue()
        threading.Thread(
            target=lambda: lines.put(self.process.stdout.readline()), daemon=True
        ).start()
        try:
            return lines.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()


class Cluster:
    def __init__(self, binary: Path, model_dir: Path, log_dir: Path, threads: int):
        self.args = (binary, model_dir, log_dir, threads)
        self.workers: list[Worker] = []

    def start(self, count: int) -> list[str]:
        self.stop()
        binary, model_dir, log_dir, threads = self.args
        names = [f"e2e-{count}-{i}" for i in range(count)]
        self.workers = [Worker(binary, model_dir, name, log_dir, threads) for name in names]
        return [worker.target for worker in self.workers]

    def stop(self) -> None:
        for worker in self.workers:
            worker.stop()
        self.workers = []


def as_rows(generation) -> np.ndarray:
    return np.frombuffer(generation.logits, dtype="<f4").reshape(generation.rows, generation.vocab)


def run(cluster, digest, prompt, gen, cuts, device, *, logits=True):
    targets = cluster.start(len(cuts) + 1)
    with Pipeline(targets, digest, cuts, device=device, step_timeout=300) as pipeline:
        pipeline.load()
        return pipeline.model.n_layer, pipeline.generate(prompt, gen, logits=logits)


def prepare_tiny(work: Path, gen: int):
    from make_tiny_llama import make

    hf_dir, gguf = make(work / "tiny")
    rows, tokens = greedy(load_model(str(hf_dir)), TINY_PROMPT, gen)
    return gguf, TINY_PROMPT, rows, tokens


def prepare_tinyllama(backend: str, gguf: Path | None):
    cache = ROOT / ".cache"
    gguf = (
        gguf
        or cache / "models" / f"tinyllama-1.1b-chat-{'f32' if backend == 'cpu' else 'f16'}.gguf"
    )
    reference = json.loads((cache / "runs" / "ref.tokens").read_text())
    rows = read_logits(cache / "runs" / "ref.bin")
    return gguf, reference["prompt"], rows, reference["generated"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", choices=sorted(DEVICES), default="cpu")
    parser.add_argument("--model", choices=["tiny", "tinyllama"], default="tiny")
    parser.add_argument("--gguf", type=Path, help="override the GGUF file of --model tinyllama")
    parser.add_argument("--worker-bin", type=Path)
    parser.add_argument("--gen", type=int, default=12, help="tokens to generate (tiny model)")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    binary = args.worker_bin or os.environ.get("SHARDLINE_WORKER_BIN")
    binary = (
        Path(binary) if binary else ROOT / "worker" / "build" / args.backend / "shardline-worker"
    )
    if not binary.exists():
        print(
            f"no worker binary at {binary}; build it or set SHARDLINE_WORKER_BIN", file=sys.stderr
        )
        return 2

    checks: list[Check] = []
    device = DEVICES[args.backend]
    with tempfile.TemporaryDirectory(prefix="shardline-e2e-") as directory:
        work = Path(directory)
        if args.model == "tiny":
            gguf, prompt, ref_rows, ref_tokens = prepare_tiny(work, args.gen)
        else:
            gguf, prompt, ref_rows, ref_tokens = prepare_tinyllama(args.backend, args.gguf)
        gen = len(ref_tokens)
        fp32 = args.model == "tiny" or gguf.name.endswith("f32.gguf")

        model_dir = work / "models"
        digest = add_model(gguf, model_dir)
        print(
            f"model {gguf.name} ({digest[:12]}), prompt of {len(prompt)} tokens, {gen} to generate"
        )

        cluster = Cluster(binary, model_dir, work, args.threads)
        try:
            n_layer, single = run(cluster, digest, prompt, gen, [], device)
            base = as_rows(single)
            checks.append(Check("one worker generates", len(single.tokens) == gen))

            splits = {"two workers": [n_layer // 2]}
            if n_layer >= 3:
                splits["three workers"] = [n_layer // 3, 2 * n_layer // 3]
            for name, cuts in splits.items():
                _, split = run(cluster, digest, prompt, gen, cuts, device)
                rows = as_rows(split)
                exact = np.array_equal(rows, base) and split.tokens == single.tokens
                checks.append(Check(f"{name}, cuts {cuts}: bit-exact with one worker", exact))

            _, quick = run(
                cluster, digest, prompt, gen, splits["two workers"], device, logits=False
            )
            checks.append(
                Check("token-only mode picks the same tokens", quick.tokens == single.tokens)
            )
        finally:
            cluster.stop()

        p = compare(ref_rows, base, len(prompt))
        detail = (
            f"top-1 {p.top1:.4f}, max |dlogit| {p.max_abs:.2e}, mean KL {p.kl_mean:.2e}, "
            f"first token divergence: {p.diverged_at}"
        )
        checks.append(
            Check("greedy tokens equal Hugging Face", single.tokens == ref_tokens, detail)
        )
        checks.append(Check(f"top-1 agreement >= {MIN_TOP1}", p.top1 >= MIN_TOP1, f"{p.top1:.4f}"))
        checks.append(
            Check(f"mean KL <= {MAX_MEAN_KL:g}", p.kl_mean <= MAX_MEAN_KL, f"{p.kl_mean:.2e}")
        )
        if fp32 and args.backend == "cpu":
            checks.append(
                Check(
                    f"max |dlogit| <= {MAX_FP32_CPU_LOGIT_DIFF:g} (fp32 on the CPU)",
                    p.max_abs <= MAX_FP32_CPU_LOGIT_DIFF,
                    f"{p.max_abs:.2e}",
                )
            )

    width = max(len(check.name) for check in checks)
    for check in checks:
        print(f"{'ok  ' if check.ok else 'FAIL'} {check.name:<{width}}  {check.detail}")
    failed = [check for check in checks if not check.ok]
    print(f"\n{len(checks) - len(failed)} of {len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
