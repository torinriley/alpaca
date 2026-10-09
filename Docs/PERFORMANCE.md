# Performance

*Author: Torin Etheridge · 2026-10-09 · MIT License*

Hardware for every number in this file: **Apple M5 (10-core GPU, Apple10 family), 16 GB, macOS 27.0, Swift 6.4 release build**, model **SmolLM2-135M-Instruct** (F16 / Q8_0 / Q4_0 GGUF), Metal backend, greedy decoding, 64 generated tokens, 7 repetitions per cell after a warm-up. Only this machine was measured; nothing here was run on an iPhone or iPad.

> **Measurement environment.** The machine was *not* idle: load average 5–7 (a browser, WindowServer driving a second virtual display, background agents) for the whole of Phase 3. Absolute numbers drift by up to ±15–25% over minutes — llama.cpp itself measured 317 → 265 tok/s on F16 decode between two runs a few hours apart. Therefore (a) every optimisation was judged by interleaved A/B runs in the same time window or by GPU-timestamp time rather than wall time, and (b) comparisons with llama.cpp are only meaningful within one run (`Scripts/compare_llamacpp.sh` runs both back to back per format). For publishable absolute numbers, rerun on an idle machine.

## Reproduce

```sh
Scripts/fetch_models.sh
Scripts/compare_llamacpp.sh Benchmarks/results/<name>    # alpaca + llama-bench over the full grid, prints the table below
.build/release/alpaca profile <model.gguf> --tokens 2048 --context 2200       # per-stage GPU time of one prefill pass
.build/release/alpaca profile <model.gguf> --tokens 1 --context-before 2048   # per-stage GPU time of one decode step
```
Raw data: `Benchmarks/results/phase3-baseline/` (before) and `phase3-final/` (after): JSON with hardware, OS, model, per-cell median/min/max/stdev, load times, peak resident and peak physical footprint, thermal state.

## Method

- `prefill tok/s` = prompt tokens ÷ time until the first token's logits exist; session creation is excluded, but sessions are reused (see below), so repeated runs measure warm sessions. `decode tok/s` = (generated−1) ÷ time between first and last token, through the public streaming API.
- Prefill prompts are the first *N* tokens of a fixed text; decode runs start from a prompt of the stated context length. EOS stopping is off. Cold load (first load in the process: shader compile + first page-in) vs warm load are reported separately.
- GPU times come from Metal command-buffer timestamps; per-stage profiles use GPU timestamps sampled at encoder boundaries (`MTLCounterSampleBuffer`, stage boundary) so stages are measured without commit/wait gaps. Profiled stages serialise and add small launch costs, so their sum can slightly exceed an unprofiled pass.
- Peak memory is `ri_lifetime_max_phys_footprint` (what iOS jetsam counts: dirty + compressed + GPU-wired memory). Mapped weights are file-backed clean pages and appear in resident size, not footprint.
- llama.cpp: Homebrew build 11429 (`d81235049`), `-ngl 99`, f16 K/V, flash-attention auto, `-r 7`, same files.

## Result: before → after (tok/s, median of 7; llama.cpp measured in the same run)

**Prefill** (prompt of N tokens from an empty context)

| format | N | baseline | **now** | llama.cpp | now / llama.cpp |
|---|---|---|---|---|---|
| Q8_0 | 128 | 5,563 | **17,351** | 17,163 | 1.01× |
| Q8_0 | 512 | 3,994 | **23,864** | 25,581 | 0.93× |
| Q8_0 | 2048 | 1,541 | **17,041** | 18,729 | 0.91× |
| Q8_0 | 4096 | 844 | **12,256** | 14,842 | 0.83× |
| Q4_0 | 128 / 2048 | 6,019 / 1,542 | **18,874 / 17,427** | 18,085 / 20,029 | 1.04× / 0.87× |
| F16 | 128 / 2048 | 5,868 / 1,543 | **19,170 / 19,153** | 15,235 / 18,160 | 1.26× / 1.05× |
| Q8_0 / Q4_0 / F16 | 16 | 1,853 / 1,942 / 1,872 | **2,850 / 3,061 / 2,901** | 4,380 / 4,140 / 3,314 | 0.65× / 0.74× / 0.88× |

**Decode** (64 tokens after a context of N tokens)

| format | context | baseline | **now** | llama.cpp | now / llama.cpp |
|---|---|---|---|---|---|
| Q4_0 | 16 | 532 | **671** | 503 | 1.33× |
| F16 | 16 | 299 | **327** | 305 | 1.07× |
| Q8_0 | 16 | 456 | **549** | 430 | 1.28× |
| Q8_0 | 512 | 330 | **469** | 401 | 1.17× |
| Q8_0 | 2048 | 177 | **412** | 354 | 1.16× |
| Q8_0 | 4096 | 109 | **378** | 322 | 1.17× |
| Q4_0 | 2048 / 4096 | 187 / 113 | **470 / 395** | 391 / 344 | 1.20× / 1.15× |

Full grids (16…4096 for all three formats) are in `Benchmarks/results/phase3-final/*.json`; baseline in `phase3-baseline/`. Stretch targets from the Phase 3 brief: prefill 128 Q8_0 ≥ 15,000 → **17,351 ✓**; prefill 2048 Q8_0 ≥ 17,000 → **17,041 ✓ (at the margin of measurement noise)**; decode 16 Q4_0 ≥ 542 → **671 ✓**; decode 16 F16 ≥ 305 → **327 ✓**; decode 2048 Q8_0 ≥ 350 → **412 ✓**. Memory: peak physical footprint **172 MiB (F16), 176 MiB (Q8_0), 192 MiB (Q4_0)** over the whole benchmark grid with a 4400-token context (the f16 KV cache alone is about 97 MiB of that).

Not met: prefill still trails llama.cpp at long prompts on quantised files (0.80–0.93× at 512–4096) and at very short prompts (16 tokens: 0.65–0.88×). Decode with a stochastic sampler (temperature > 0) does not get the GPU-chained path (see below) and runs ~15–20% slower than greedy.

## Where the time went (profiles, Q8_0)

**Baseline prefill, 2048 tokens (1,270 ms GPU):** attention **87.5%** (1,111 ms), FFN gate+up 5.5%, FFN down 2.9%, Q/K/V 2.0%, everything else < 2% each. The scalar attention kernel did O(n²) work with one threadgroup per (head, token) and re-read each K/V row once per query head.
**Baseline decode, context 2048 (5.5 ms/token):** attention **65%** (3.6 ms) reading the KV cache at ~13 GB/s because a layer has only `heads` = 9 units of work, FFN gate+up 10%, Q/K/V 5%. At context 16 (1.7 ms/token) attention is 6% and the big matrices run at 100–125 GB/s — decode is already near the memory limit there, and ~20% of GPU time was ~390 tiny dispatches (norm, RoPE, residual, SiLU, 13 per layer).
**After prefill, 2048 tokens (≈ 125–135 ms):** attention 43–45%, FFN gate+up+SiLU 20–24%, FFN down 10–12%, Q/K/V 7%, residual-fused out-projection 4–5%, RMSNorm 2–5%, RoPE 2–3%.

## Optimisation log

Every row has a baseline, a correctness check (Docs/NUMERICAL_VALIDATION.md) and a measurement. "A/B" = alternating runs in one window. Numbers are Q8_0 unless stated.

| # | change | measured effect |
|---|---|---|
| 1 | Prefill chunk 128 → 512 tokens (fewer weight re-reads) | prefill@512 4,039 → 4,453; @2048 1,543 → 1,611 (old kernels) |
| 2 | **Tiled causal attention** (FlashAttention-style, `simdgroup_matrix`, online softmax, half Q/K/V/P, float32 accumulate; causal block skipping) | prefill@2048 1,610 → 8,543; attention 1,111 → ~130 ms |
| 3 | **Metal 4 tensor-op GEMM** (`matmul2d`, relaxed precision) for batches ≥ 32 → ≥ 16; quantised weights expanded one matrix at a time into a reusable half scratch | prefill@128 7.7k → 11.7k, @512 10.3k → 18.1k, @2048 8.6k → 13.5k; threshold 32 → 16: 16-token prefill 2,374 → 3,166, 24-token 2,541 → 4,440 |
| 4 | Attention softmax with two lanes per row (no 5-stage shuffle reductions) and `fast::exp` | attention kernel 2.17 → 1.61 ms per layer-call (−26%), prefill@2048 13.5k → 14.8k |
| 5 | Session reuse (freed KV/scratch buffers are recycled; a fresh session costs buffer allocation + first-touch page faults) and residual add fused into the projection (`C += A·B`, one dispatch, no temp buffer) | prefill@128 ~11k → 17.7k (cold-session overhead removed), @2048 14.8k → 16.1k |
| 6 | Fused gate + up + SiLU·mul tensor GEMM (both products stay in registers; neither activation is written) | GPU time of a 2048 prefill −4…−10% (A/B) |
| 7 | **Split-K decode attention with GQA sharing** (one threadgroup per KV head × position split; K/V read once for the 3 query heads that share them; quad-per-position layout; adaptive ~8 splits) | decode@2048 157 → 310 tok/s (A/B), @4096 ~100 → 271 |
| 8 | Fused decode kernels: RMSNorm in the mat-vec prologue, Q/K/V as one dispatch, gate+up+SiLU as one, RoPE+KV-store as one (13 → 6 dispatches/layer) | GPU time/token Q8_0 −6% (ctx16) / −2% (ctx2048); Q4_0 −5% / −4%; F16 −15% / −9% |
| 9 | **GPU-resident greedy decoding**: arg-max kernel feeds the next step directly; 3 command buffers in flight, CPU only streams tokens | Q8_0 437 → 521 (ctx16), 345 → 397 (ctx2048); Q4_0 520 → 646, 394 → 464; F16 270 → 299, 234 → 255 (A/B) |
| 10 | Memory audit of this phase's own changes: the first version of the half-precision dequant scratch was sized for the largest matrix *including* the 28M-element output/embedding table (113 MiB wasted per session, found when the memory estimate jumped to 443 MiB); that matrix is only ever a mat-vec, so it is now excluded. The unused temporary output buffer was removed | dequant scratch 113 MiB → 3.5 MiB. Net effect against the Phase 2 baseline: estimated scratch 3.3 → 16.0 MiB (larger prefill chunk + two dequant scratches), i.e. +12.7 MiB bought for the prefill speed-ups |

### Tried and rejected (measured, kept out of the code)

- **Half-precision operands in the simdgroup-matrix GEMM** (instead of float32): +1% — the GEMM is not threadgroup-bandwidth bound. Reverted to keep exact float32 operands in the `.exact` path.
- **Event-gated pre-encoded decode steps** (encode the next command buffer while sampling): no measurable gain (±1%); removed. Decode time per token includes a CPU round trip that only *not having one* (item 9) removes: back-to-back committed steps were 6–14% faster than synchronous ones in a controlled experiment.
- **Tensor-op (neural-accelerator) attention** (QKᵀ and PV as `matmul2d` on 64×32 tiles, softmax in threadgroup memory): prototype 1.84 ms per layer-call vs 1.61 ms for the shipped simdgroup-matrix kernel, even without softmax (1.75 ms): tile sizes small enough for threadgroup memory are barrier-bound. The simdgroup kernel runs at ~3 TFLOPs, close to simdgroup-matrix peak.
- **Larger prefill chunks (1024, 2048)**: equal or slower than 512 (cache locality), despite fewer weight dequantisations.
- **Decode split length** (64/128/256/512 fixed, adaptive 4/8/16 splits): fixed splits trade short vs long context; ~8 splits per KV head won everywhere.

## Remaining bottlenecks and recommended next work

1. **Prefill attention is 43–45% of a long prefill and sits at simdgroup-matrix peak** (~3 TFLOPs). Closing the last 10–20% against llama.cpp at 2–4k tokens likely needs a different structure (query tile shared across the 3 GQA heads, 64-position blocks, or tensor ops with larger tiles than threadgroup memory allows).
2. **Short prompts (≤ 24 tokens)** are 1–3 weight passes of mat-vec kernels; a dedicated small-batch kernel (several tokens per weight read with the fused decode prologue) could roughly halve them.
3. **Sampling on the GPU for temperature/top-k/top-p** (Gumbel-max + partial selection) would extend the chained path beyond greedy; stochastic decoding currently pays ~0.3 ms/token CPU round trip.
4. **Memory-bandwidth headroom:** at short context, decode streams weights at ~100–125 GB/s; the device bandwidth was not measured separately, so the remaining headroom is unknown.
5. **Everything above is a single-device, single-model measurement.** iPhone/iPad GPUs without the M5 tensor hardware fall back to the `.exact` simdgroup GEMM (roughly half the `.fast` prefill throughput measured here: 7.7–10.3k vs 11.7–18.1k tok/s at 128–512 tokens) — unmeasured on those devices. Energy and thermal behaviour under sustained load are unmeasured.
