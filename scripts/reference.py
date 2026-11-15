"""HF transformers fp32 CPU reference: greedy decode with every logits row saved.

    uv run --extra oracle python scripts/reference.py --gen 32 --out .cache/runs/ref
    uv run --extra oracle python scripts/reference.py --model DIR --tokens 1,2,3 --out ref

Writes <out>.tokens (the prompt ids and the generated ids) and <out>.bin in the
layout `shardline generate --logits-out` uses: one logits row per prompt
position, then one per decode step.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from compare import write_logits

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
PROMPT = "Explain in two sentences why the sky is blue."


def prompt_tokens(tokenizer) -> list[int]:
    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids if ids[0] == tokenizer.bos_token_id else [tokenizer.bos_token_id, *ids]


def load_model(model: str, revision: str | None = None):
    return AutoModelForCausalLM.from_pretrained(
        model, revision=revision, dtype=torch.float32, attn_implementation="eager"
    ).eval()


def greedy(model, prompt: list[int], gen: int) -> tuple[np.ndarray, list[int]]:
    """Greedy decoding with a KV cache. Returns every logits row and the generated ids."""
    rows = []
    generated = []
    with torch.no_grad():
        output = model(torch.tensor([prompt]), use_cache=True)
        rows.extend(output.logits[0].numpy())
        past = output.past_key_values
        next_token = int(output.logits[0, -1].argmax())
        for step in range(gen):
            generated.append(next_token)
            if step + 1 == gen:
                break
            output = model(torch.tensor([[next_token]]), past_key_values=past, use_cache=True)
            past = output.past_key_values
            rows.append(output.logits[0, -1].numpy())
            next_token = int(output.logits[0, -1].argmax())
    return np.stack(rows), generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL, help="a Hugging Face id or a local directory")
    parser.add_argument("--revision", help="default: the pinned one for the default model")
    parser.add_argument("--tokens", help="comma-separated prompt ids; skips the tokenizer")
    parser.add_argument("--gen", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    revision = args.revision or (REVISION if args.model == MODEL else None)
    tokenizer = None
    if args.tokens:
        prompt = [int(token) for token in args.tokens.split(",")]
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
        prompt = prompt_tokens(tokenizer)

    rows, generated = greedy(load_model(args.model, revision), prompt, args.gen)

    out = Path(args.out)
    write_logits(out.with_suffix(".bin"), rows)
    out.with_suffix(".tokens").write_text(json.dumps({"prompt": prompt, "generated": generated}))
    print("prompt:", ",".join(map(str, prompt)))
    print("generated:", ",".join(map(str, generated)))
    if tokenizer is not None:
        print("text:", tokenizer.decode(generated))


if __name__ == "__main__":
    main()
