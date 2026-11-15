"""Compares two logits dumps (same layout as stage_probe and reference.py).

    python spike/compare.py REF.bin TEST.bin [--prompt-len N]

Rows are compared only while both runs fed the same tokens, i.e. up to and
including the first row where their greedy choices differ.
"""

import argparse

import numpy as np


def read_logits(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        rows, cols = np.fromfile(f, dtype=np.int32, count=2)
        return np.fromfile(f, dtype=np.float32).reshape(rows, cols)


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("test")
    parser.add_argument("--prompt-len", type=int, required=True)
    args = parser.parse_args()

    ref, test = read_logits(args.reference), read_logits(args.test)
    rows = min(len(ref), len(test))
    ref_top, test_top = ref[:rows].argmax(-1), test[:rows].argmax(-1)

    # Generated tokens come from the last prompt row onwards.
    gen_ref, gen_test = ref_top[args.prompt_len - 1 :], test_top[args.prompt_len - 1 :]
    pairs = enumerate(zip(gen_ref, gen_test, strict=True))
    diverge = next((i for i, (a, b) in pairs if a != b), None)
    usable = rows if diverge is None else args.prompt_len + diverge

    r, t = ref[:usable], test[:usable]
    lp_ref, lp_test = log_softmax(r), log_softmax(t)
    kl = (np.exp(lp_ref) * (lp_ref - lp_test)).sum(-1)
    bit_exact = bool(np.array_equal(ref[:rows], test[:rows]))

    print(f"rows compared        {usable} of {rows}")
    print(f"bit-exact            {bit_exact}")
    print(
        f"greedy tokens equal  {diverge is None} ({len(gen_ref)} generated)"
        + ("" if diverge is None else f", first divergence at step {diverge}")
    )
    print(f"top-1 agreement      {(r.argmax(-1) == t.argmax(-1)).mean():.4f}")
    print(f"max |dlogit|         {np.abs(r - t).max():.3e}")
    print(f"mean |dlogit|        {np.abs(r - t).mean():.3e}")
    print(f"KL(ref||test) mean   {kl.mean():.3e}  max {kl.max():.3e}")


if __name__ == "__main__":
    main()
