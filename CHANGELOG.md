# Changelog

*Author: Torin Etheridge · 2026-10-09 · MIT License*

## Unreleased

Prefill (branch `perf/quantized-prefill`): half-precision activations into the tensor-op GEMM, tensor-op causal attention with a swept tile shape, vectorised weight expansion. Q8_0 prefill at 2048 tokens 17.0k → 30.8k tok/s, at 4096 tokens 12.3k → 24.1k; prefill is now 1.3–1.7× llama.cpp from 128 to 4096 tokens (still behind for ≤ 24-token prompts).

## 0.1.0 — 2026-10-09 (pre-release)

Performance (Phase 3): tiled causal prefill attention, Metal 4 tensor-op GEMM (`GEMMPrecision.fast`), fused gate/up/SiLU and residual-accumulating projections, split-K GQA decode attention, fused single-token kernels, GPU-resident greedy decoding with command buffers in flight, session reuse, `alpaca profile`. Prefill 128 tokens Q8_0 5.6k → 17.4k tok/s, 2048 tokens 1.5k → 17.0k; decode at 2048 context Q8_0 177 → 412 tok/s. See Docs/PERFORMANCE.md.
Behaviour changes: default prefill batch 512; default prefill projections are half-precision on Apple10-family GPUs (`LoadOptions.prefillPrecision = .exact` to disable); `LanguageModel.trimMemory()` added.

- Tensor engine, CPU reference operations, quantised Q8_0/Q4_0 blocks.
- Llama decoder (GQA, RoPE, SwiGLU, KV cache) on CPU and Metal.
- GGUF v2/v3 parser with bounds-checked, fuzzed parsing; byte-level BPE tokenizer.
- Async streaming public API, seeded sampling, cancellation, memory budgeting.
- `alpaca` CLI (`info`, `generate`, `bench`).
