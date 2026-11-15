"""HF transformers fp32 CPU reference: greedy decode with every logits row saved.

    uv run --extra oracle python spike/reference.py --gen 32 --out .cache/ref

Writes <out>.tokens (prompt ids, then generated ids) and <out>.bin in the
layout stage_probe uses: one logits row per prompt position, then one per
decode step.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
PROMPT = "Explain in two sentences why the sky is blue."


def prompt_tokens(tokenizer) -> list[int]:
    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids if ids[0] == tokenizer.bos_token_id else [tokenizer.bos_token_id, *ids]


def write_logits(path: Path, rows: list[np.ndarray]) -> None:
    data = np.stack(rows).astype(np.float32)
    with path.open("wb") as f:
        np.array(data.shape, dtype=np.int32).tofile(f)
        data.tofile(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    prompt = prompt_tokens(tokenizer)
    rows = []
    generated = []
    with torch.no_grad():
        output = model(torch.tensor([prompt]), use_cache=True)
        rows.extend(output.logits[0].numpy())
        past = output.past_key_values
        next_token = int(output.logits[0, -1].argmax())
        for step in range(args.gen):
            generated.append(next_token)
            if step + 1 == args.gen:
                break
            output = model(torch.tensor([[next_token]]), past_key_values=past, use_cache=True)
            past = output.past_key_values
            rows.append(output.logits[0, -1].numpy())
            next_token = int(output.logits[0, -1].argmax())

    out = Path(args.out)
    write_logits(out.with_suffix(".bin"), rows)
    out.with_suffix(".tokens").write_text(json.dumps({"prompt": prompt, "generated": generated}))
    print("prompt:", ",".join(map(str, prompt)))
    print("generated:", ",".join(map(str, generated)))
    print("text:", tokenizer.decode(generated))


if __name__ == "__main__":
    main()
