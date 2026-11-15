"""Builds a small random Llama: an HF model to use as the oracle, and the same weights as GGUF.

    uv run --extra oracle python scripts/make_tiny_llama.py --out .cache/tiny

Writes OUT/hf/ and OUT/tiny.gguf. The weights are random, so only the arithmetic is
tested, but it needs no download and takes a second. The GGUF layout follows
llama.cpp's convert_hf_to_gguf.py.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter
from transformers import LlamaConfig, LlamaForCausalLM

LAYERS = 4
HIDDEN = 64
HEADS = 4
KV_HEADS = 2
FFN = 128
VOCAB = 96
CONTEXT = 128
ROPE_THETA = 10000.0
RMS_EPS = 1e-5


def build_model(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=FFN,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        max_position_embeddings=CONTEXT,
        rms_norm_eps=RMS_EPS,
        rope_parameters={"rope_type": "default", "rope_theta": ROPE_THETA},
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    # The default initialisation is so small that the logits are nearly flat. Larger
    # weights give a clear winner at every step, so a greedy comparison means something.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight"):
                parameter.copy_(1.0 + 0.1 * torch.randn_like(parameter))
            else:
                parameter.normal_(0.0, 0.25)
    return model.eval()


def permute(weight: np.ndarray, n_head: int) -> np.ndarray:
    """HF rotates halves of a head; ggml's RoPE rotates neighbouring pairs."""
    shape = weight.shape
    return (
        weight.reshape(n_head, 2, shape[0] // n_head // 2, *shape[1:]).swapaxes(1, 2).reshape(shape)
    )


def write_gguf(model: LlamaForCausalLM, path: Path) -> None:
    state = {name: tensor.detach().float().numpy() for name, tensor in model.state_dict().items()}
    writer = GGUFWriter(str(path), "llama")
    writer.add_context_length(CONTEXT)
    writer.add_embedding_length(HIDDEN)
    writer.add_block_count(LAYERS)
    writer.add_feed_forward_length(FFN)
    writer.add_head_count(HEADS)
    writer.add_head_count_kv(KV_HEADS)
    writer.add_layer_norm_rms_eps(RMS_EPS)
    writer.add_rope_freq_base(ROPE_THETA)

    writer.add_tensor("token_embd.weight", state["model.embed_tokens.weight"])
    writer.add_tensor("output_norm.weight", state["model.norm.weight"])
    writer.add_tensor("output.weight", state["lm_head.weight"])
    for i in range(LAYERS):
        prefix = f"model.layers.{i}."
        layer = {
            "attn_norm": state[prefix + "input_layernorm.weight"],
            "attn_q": permute(state[prefix + "self_attn.q_proj.weight"], HEADS),
            "attn_k": permute(state[prefix + "self_attn.k_proj.weight"], KV_HEADS),
            "attn_v": state[prefix + "self_attn.v_proj.weight"],
            "attn_output": state[prefix + "self_attn.o_proj.weight"],
            "ffn_norm": state[prefix + "post_attention_layernorm.weight"],
            "ffn_gate": state[prefix + "mlp.gate_proj.weight"],
            "ffn_up": state[prefix + "mlp.up_proj.weight"],
            "ffn_down": state[prefix + "mlp.down_proj.weight"],
        }
        for name, tensor in layer.items():
            writer.add_tensor(f"blk.{i}.{name}.weight", np.ascontiguousarray(tensor))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def make(out: Path, seed: int = 0) -> tuple[Path, Path]:
    """Returns the HF directory and the GGUF file."""
    out.mkdir(parents=True, exist_ok=True)
    model = build_model(seed)
    hf_dir = out / "hf"
    model.save_pretrained(hf_dir)
    gguf_path = out / "tiny.gguf"
    write_gguf(model, gguf_path)
    return hf_dir, gguf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    hf_dir, gguf_path = make(args.out, args.seed)
    print(f"hf:   {hf_dir}")
    print(f"gguf: {gguf_path} ({gguf_path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
