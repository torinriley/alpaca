#!/usr/bin/env python3
# Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
# Author: Torin Etheridge · 2026-10-09
"""Level 2/3 reference: tiny randomly initialised Llama models run in float64 with Hugging Face transformers.

Writes Tests/AlpacaCoreTests/Fixtures/tiny_llama.json with, per model: config, weights in HF layout
(split-half RoPE) and in GGUF layout (q/k rows permuted exactly as llama.cpp's converter does, interleaved RoPE),
input tokens and float64 logits for every position.
"""
import json, pathlib, torch, numpy as np
from transformers import LlamaConfig, LlamaForCausalLM

def permute(w, n_head, n_head_kv=None):
    if n_head_kv is not None and n_head != n_head_kv: n_head = n_head_kv
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)

def build(name, seed, heads, kv, tie):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                      num_attention_heads=heads, num_key_value_heads=kv, max_position_embeddings=64,
                      rms_norm_eps=1e-5, rope_theta=10000.0, tie_word_embeddings=tie, attention_bias=False, mlp_bias=False)
    m = LlamaForCausalLM(cfg).double().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n: p.copy_(1 + 0.2 * torch.randn_like(p))      # non-trivial norm gains
            else: p.copy_(torch.randn_like(p) * 0.15)
    tokens = torch.tensor([[5, 17, 3, 42, 63, 0, 11]])
    with torch.no_grad(): logits = m(tokens).logits[0]
    sd = {k: v.detach().numpy() for k, v in m.state_dict().items()}
    hd = 32 // heads
    def pack(gguf_layout):
        w = {"token_embd": sd["model.embed_tokens.weight"], "output_norm": sd["model.norm.weight"]}
        if not tie: w["output"] = sd["lm_head.weight"]
        for i in range(2):
            p = f"model.layers.{i}."
            q, k = sd[p + "self_attn.q_proj.weight"], sd[p + "self_attn.k_proj.weight"]
            if gguf_layout: q, k = permute(q, heads), permute(k, heads, kv)
            w.update({f"blk.{i}.attn_norm": sd[p + "input_layernorm.weight"], f"blk.{i}.attn_q": q, f"blk.{i}.attn_k": k,
                      f"blk.{i}.attn_v": sd[p + "self_attn.v_proj.weight"], f"blk.{i}.attn_output": sd[p + "self_attn.o_proj.weight"],
                      f"blk.{i}.ffn_norm": sd[p + "post_attention_layernorm.weight"], f"blk.{i}.ffn_gate": sd[p + "mlp.gate_proj.weight"],
                      f"blk.{i}.ffn_up": sd[p + "mlp.up_proj.weight"], f"blk.{i}.ffn_down": sd[p + "mlp.down_proj.weight"]})
        return {k: {"shape": list(v.shape), "data": v.reshape(-1).tolist()} for k, v in w.items()}
    return dict(config=dict(vocab=64, hidden=32, layers=2, heads=heads, kvHeads=kv, headDim=hd, ffn=48, context=64,
                            eps=1e-5, theta=10000.0, tied=tie),
                tokens=tokens[0].tolist(), logits=logits.reshape(-1).tolist(),
                weights_hf=pack(False), weights_gguf=pack(True))

out = {"gqa_untied": build("gqa_untied", 1, 4, 2, False), "mha_tied": build("mha_tied", 2, 4, 4, True)}
p = pathlib.Path(__file__).resolve().parent.parent / "Tests/AlpacaCoreTests/Fixtures/tiny_llama.json"
p.write_text(json.dumps(out))
print("wrote", p, p.stat().st_size)
