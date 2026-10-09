#!/usr/bin/env python3
# Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
# Author: Torin Etheridge · 2026-10-09
"""Generates independent reference values for the Level-1 operation tests.

References are computed with numpy (float64) / PyTorch / the `gguf` package — never with alpaca code.
Output: Tests/AlpacaCoreTests/Fixtures/ops.json
Run:  .venv/bin/python Scripts/generate_op_fixtures.py
"""
import json, base64, pathlib
import numpy as np, torch
from gguf.quants import quantize as gguf_quantize, dequantize as gguf_dequantize
from gguf import GGMLQuantizationType as QT

rng = np.random.default_rng(20260101)
f = lambda a: np.asarray(a, dtype=np.float64).reshape(-1).tolist()
out = {}

def r(*shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32)

# Elementwise
a, b = r(3, 5), r(3, 5)
out["add"] = dict(shape=[3, 5], a=f(a), b=f(b), expected=f(a.astype(np.float64) + b))
out["mul"] = dict(shape=[3, 5], a=f(a), b=f(b), expected=f(a.astype(np.float64) * b))

x = r(4, 9, scale=3)
out["silu"] = dict(shape=[4, 9], x=f(x), expected=f(torch.nn.functional.silu(torch.tensor(x, dtype=torch.float64)).numpy()))

# Matmul
A, B = r(4, 7), r(7, 5)
out["matmul"] = dict(m=4, k=7, n=5, a=f(A), b=f(B), expected=f(A.astype(np.float64) @ B.astype(np.float64)))

# Linear (x @ W^T), f32 and f16 weights
X, W = r(3, 64), r(10, 64, scale=0.2)
W16 = W.astype(np.float16)
out["linear"] = dict(m=3, k=64, n=10, x=f(X), w=f(W), expected=f(X.astype(np.float64) @ W.astype(np.float64).T),
                     w_f16=f(W16.astype(np.float64)), expected_f16=f(X.astype(np.float64) @ W16.astype(np.float64).T))

# RMSNorm (matches Llama: x * rsqrt(mean(x^2)+eps) * w)
Xn, Wn = r(3, 16, scale=2), r(16)
Xt = torch.tensor(Xn, dtype=torch.float64)
rms = Xt * torch.rsqrt(Xt.pow(2).mean(-1, keepdim=True) + 1e-5) * torch.tensor(Wn, dtype=torch.float64)
out["rmsnorm"] = dict(shape=[3, 16], eps=1e-5, x=f(Xn), w=f(Wn), expected=f(rms.numpy()))

# Softmax incl. large magnitudes (stability)
Xs = r(3, 11, scale=30)
out["softmax"] = dict(shape=[3, 11], x=f(Xs), expected=f(torch.softmax(torch.tensor(Xs, dtype=torch.float64), -1).numpy()))

# RoPE
T, H, D, start, theta = 5, 3, 16, 7, 100000.0
Xr = r(T, H, D)
pos = (np.arange(T) + start).astype(np.float64)[:, None]
inv = theta ** (-2.0 * np.arange(D // 2) / D)
ang = (pos * inv[None, :])[:, None, :]  # [T,1,D/2]
z = Xr[..., 0::2].astype(np.float64) + 1j * Xr[..., 1::2].astype(np.float64)
zr = z * np.exp(1j * ang)
inter = np.empty_like(Xr, dtype=np.float64); inter[..., 0::2] = zr.real; inter[..., 1::2] = zr.imag
x1, x2 = Xr[..., : D // 2].astype(np.float64), Xr[..., D // 2:].astype(np.float64)
c, s = np.cos(ang), np.sin(ang)
split = np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)  # HF rotate_half convention
out["rope"] = dict(shape=[T, H, D], start=start, theta=theta, x=f(Xr), expected_interleaved=f(inter), expected_split=f(split))

# GQA causal attention (cache longer than needed to exercise `length`)
TQ, NH, NKV, HD, START = 3, 4, 2, 8, 2
LEN, CAP = START + TQ, 8
Q, K, V = r(TQ, NH, HD), r(CAP, NKV, HD), r(CAP, NKV, HD)
Qt = torch.tensor(Q, dtype=torch.float64).permute(1, 0, 2)                       # [H,T,D]
Kt = torch.tensor(K[:LEN], dtype=torch.float64).repeat_interleave(NH // NKV, 1).permute(1, 0, 2)
Vt = torch.tensor(V[:LEN], dtype=torch.float64).repeat_interleave(NH // NKV, 1).permute(1, 0, 2)
mask = torch.tril(torch.ones(LEN, LEN, dtype=torch.bool))[START:START + TQ]      # [T,LEN]
o = torch.nn.functional.scaled_dot_product_attention(Qt, Kt, Vt, attn_mask=mask)  # [H,T,D]
out["attention"] = dict(tokens=TQ, heads=NH, kvHeads=NKV, headDim=HD, start=START, length=LEN, capacity=CAP,
                        q=f(Q), k=f(K), v=f(V), expected=f(o.permute(1, 0, 2).numpy()))

# Quantisation: blocks produced by the gguf package (independent of alpaca's quantiser)
def quant_case(qt, name):
    k, n = 96, 5
    Wq = r(n, k, scale=0.5)
    blocks = gguf_quantize(Wq, qt)                       # uint8 [n, k/32*blockBytes]
    deq = gguf_dequantize(blocks, qt).astype(np.float64)  # [n,k]
    xq = r(2, k)
    out[name] = dict(k=k, n=n, m=2, blocks=base64.b64encode(blocks.tobytes()).decode(), dequantized=f(deq),
                     x=f(xq), expected=f(xq.astype(np.float64) @ deq.T), original=f(Wq))
quant_case(QT.Q8_0, "q8_0")
quant_case(QT.Q4_0, "q4_0")

# Embedding
tab = r(12, 8)
out["embedding"] = dict(vocab=12, dim=8, table=f(tab), tokens=[3, 0, 11], expected=f(tab[[3, 0, 11]]))

p = pathlib.Path(__file__).resolve().parent.parent / "Tests/AlpacaCoreTests/Fixtures/ops.json"
p.write_text(json.dumps(out))
print("wrote", p, p.stat().st_size, "bytes")
