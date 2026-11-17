# Changelog

*Author: Torin Etheridge · 2026-10-09 · MIT License*

## Unreleased

- **Llama 3.x (3.1 / 3.2) support:** RoPE frequency scaling from `rope_freqs.weight`, the `llama-bpe` pre-tokenizer (digit groups, case-insensitive contractions, newline handling, whole-word lookup before merges, BOS by default), and per-operation fallback in single-token decode so a feed-forward width above 4096 no longer drops the whole layer off the fused path. Verified token-for-token against llama.cpp on Llama-3.2-1B-Instruct Q4_K_M, including a 3,499-token prompt (where the scaling matters). New `alpaca tokenize` command.
- **More weight formats, all native on CPU and Metal:** Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and BF16 (Q4_1 is no longer expanded to F16 at load). See [Docs/FORMATS.md](Docs/FORMATS.md).
- **Faster short prompts:** a direct simdgroup-matrix kernel for 2–16 tokens replaces the mat-vec / tensor-op path there (16-token prefill about 1.3× faster on an M5; the tensor-op path now starts at 17 tokens).
- **Opt-in prompt-lookup speculative decoding** for greedy generation (`GenerationConfiguration.lookupDraftTokens`, `--speculate N`); output is identical to plain greedy decoding.

## 0.2.0 — 2026-11-10 (pre-release)

### Performance: GEMM Scaling

Prefill performance now scales to larger models (SmolLM2-360M, 1.7B):
- **Adaptive GEMM tile size:** 128×64 tile for large matrices (1.7B), 64×64 for small models
- **Shared-weight launch order:** Concurrent threadgroups share weight tiles, improving cache utilization
- **Optional prefill weight cache:** `--prefill-cache` flag maintains half-precision copies of quantized weights (+2 bytes per weight), eliminating per-chunk expansion overhead for Q8_0/Q4_0 on large models

**Impact:** SmolLM2-1.7B prefill with `--prefill-cache` is now 1.1–1.3× faster than llama.cpp at 128–4096 tokens (previously slower). Small models (135M, 360M) maintain their performance advantage.

### Documentation

- **Technical report:** [paper/alpaca-swift.pdf](paper/alpaca-swift.pdf) — comprehensive evaluation at 135M, 360M, and 1.7B parameters across F16, Q8_0, and Q4_0 formats with detailed GEMM scaling analysis
- **Evaluation harness:** `Experiments/AlpacaEval` for systematic benchmarking across model sizes
- Added scaling study results and graphs to README
- GitHub Sponsors funding configuration

### Community

- Swift Package Index configuration (`.spi.yml`) to restrict builds to Apple platforms
- Status badges in README (CI, release, license, Swift versions)

### Breaking changes

None — this release is backward compatible with 0.1.0.

## Unreleased

(No unreleased changes)

## 0.1.0 — 2026-10-09 (pre-release)

Performance (Phase 3): tiled causal prefill attention, Metal 4 tensor-op GEMM (`GEMMPrecision.fast`), fused gate/up/SiLU and residual-accumulating projections, split-K GQA decode attention, fused single-token kernels, GPU-resident greedy decoding with command buffers in flight, session reuse, `alpaca profile`. Prefill 128 tokens Q8_0 5.6k → 17.4k tok/s, 2048 tokens 1.5k → 17.0k; decode at 2048 context Q8_0 177 → 412 tok/s. See Docs/PERFORMANCE.md.
Behaviour changes: default prefill batch 512; default prefill projections are half-precision on Apple10-family GPUs (`LoadOptions.prefillPrecision = .exact` to disable); `LanguageModel.trimMemory()` added.

- Tensor engine, CPU reference operations, quantised Q8_0/Q4_0 blocks.
- Llama decoder (GQA, RoPE, SwiGLU, KV cache) on CPU and Metal.
- GGUF v2/v3 parser with bounds-checked, fuzzed parsing; byte-level BPE tokenizer.
- Async streaming public API, seeded sampling, cancellation, memory budgeting.
- `alpaca` CLI (`info`, `generate`, `bench`).
