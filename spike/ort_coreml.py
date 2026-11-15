"""Times greedy decoding of the Optimum ONNX export with ONNX Runtime providers.

    uv run --no-project --with optimum-onnx --with 'onnxruntime<1.32' \
        python spike/ort_coreml.py .cache/onnx/tinyllama
"""

import sys
import time

import onnxruntime as ort
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

PROMPT = "Explain in two sentences why the sky is blue."


def run(path: str, provider: str, options: dict | None = None) -> None:
    tokenizer = AutoTokenizer.from_pretrained(path)
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 1
    start = time.perf_counter()
    model = ORTModelForCausalLM.from_pretrained(
        path, provider=provider, provider_options=options, session_options=session_options
    )
    load_s = time.perf_counter() - start
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    model.generate(**inputs, max_new_tokens=4, do_sample=False)
    for new_tokens in (1, 65):
        start = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)
        elapsed = time.perf_counter() - start
        if new_tokens == 1:
            first = elapsed
    decode = (elapsed - first) / 64
    print(
        f"{provider}: load {load_s:.1f} s, first token {first * 1000:.0f} ms, "
        f"decode {decode * 1000:.1f} ms/token = {1 / decode:.1f} t/s, "
        f"providers in session: {model.model.get_providers()}"
    )
    print("  text:", tokenizer.decode(out[0, inputs["input_ids"].shape[1] :][:24]))


if __name__ == "__main__":
    path = sys.argv[1]
    run(path, "CPUExecutionProvider")
    run(path, "CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"})
