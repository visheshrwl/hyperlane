"""Compares two logits dumps (the layout `shardline generate --logits-out` writes).

    python scripts/compare.py REF.bin TEST.bin --prompt-len N

Rows are compared only while both runs fed the same tokens, i.e. up to and
including the first row where their greedy choices differ.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def read_logits(path: str | Path) -> np.ndarray:
    with open(path, "rb") as f:
        rows, cols = np.fromfile(f, dtype=np.int32, count=2)
        return np.fromfile(f, dtype=np.float32).reshape(rows, cols)


def write_logits(path: str | Path, rows: np.ndarray) -> None:
    data = np.ascontiguousarray(rows, dtype=np.float32)
    with open(path, "wb") as f:
        np.array(data.shape, dtype=np.int32).tofile(f)
        data.tofile(f)


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


@dataclass(frozen=True)
class Parity:
    rows: int  # rows both dumps have
    usable: int  # rows compared: those fed the same tokens
    bit_exact: bool  # every row, bit for bit
    diverged_at: int | None  # first generated step whose greedy choices differ
    generated: int
    top1: float  # share of compared rows with the same argmax
    max_abs: float
    mean_abs: float
    kl_mean: float  # KL(reference || test) per row
    kl_max: float

    @property
    def same_tokens(self) -> bool:
        return self.diverged_at is None


def compare(ref: np.ndarray, test: np.ndarray, prompt_len: int) -> Parity:
    rows = min(len(ref), len(test))
    ref_top, test_top = ref[:rows].argmax(-1), test[:rows].argmax(-1)

    # Generated tokens come from the last prompt row onwards.
    gen_ref, gen_test = ref_top[prompt_len - 1 :], test_top[prompt_len - 1 :]
    pairs = enumerate(zip(gen_ref, gen_test, strict=True))
    diverge = next((i for i, (a, b) in pairs if a != b), None)
    usable = rows if diverge is None else prompt_len + diverge

    r, t = ref[:usable], test[:usable]
    lp_ref, lp_test = log_softmax(r), log_softmax(t)
    kl = (np.exp(lp_ref) * (lp_ref - lp_test)).sum(-1)
    return Parity(
        rows=rows,
        usable=usable,
        bit_exact=bool(np.array_equal(ref[:rows], test[:rows])),
        diverged_at=diverge,
        generated=len(gen_ref),
        top1=float((r.argmax(-1) == t.argmax(-1)).mean()),
        max_abs=float(np.abs(r - t).max()),
        mean_abs=float(np.abs(r - t).mean()),
        kl_mean=float(kl.mean()),
        kl_max=float(kl.max()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("test")
    parser.add_argument("--prompt-len", type=int, required=True)
    args = parser.parse_args()

    p = compare(read_logits(args.reference), read_logits(args.test), args.prompt_len)
    print(f"rows compared        {p.usable} of {p.rows}")
    print(f"bit-exact            {p.bit_exact}")
    print(
        f"greedy tokens equal  {p.same_tokens} ({p.generated} generated)"
        + ("" if p.same_tokens else f", first divergence at step {p.diverged_at}")
    )
    print(f"top-1 agreement      {p.top1:.4f}")
    print(f"max |dlogit|         {p.max_abs:.3e}")
    print(f"mean |dlogit|        {p.mean_abs:.3e}")
    print(f"KL(ref||test) mean   {p.kl_mean:.3e}  max {p.kl_max:.3e}")


if __name__ == "__main__":
    main()
