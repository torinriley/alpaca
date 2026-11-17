# Features added on this branch

*Author: Torin Etheridge · MIT License*

This page lists what the branch adds, how to use it, how it was checked, and what is **not** done. Numbers are from one Apple M5 (16 GB) that was
not idle (load average 3–7), so treat every speed figure as ±10%. Comparisons with llama.cpp (Homebrew build 11429, Metal, `-ngl 99`) were taken
back-to-back on the same machine and the same GGUF file.

## Summary

| # | Feature | Status | Where to read more |
|---|---|---|---|
| 1 | Nine more weight formats: Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, BF16 | done, validated against llama.cpp | [FORMATS.md](FORMATS.md) |
| 2 | Llama 3.1 / 3.2 support (RoPE frequency scaling, `llama-bpe` tokenizer) | done, validated against llama.cpp | below |
| 3 | Faster Q4_K / Q6_K single-token decode | done | below |
| 4 | Fused dequantisation in the tensor-op prefill GEMM | done (applies to large matrices, ≤ 192 tokens) | below |
| 5 | Small-batch kernel for 2–16 tokens | done (short prompts, speculative verification) | below |
| 6 | Prompt-lookup speculative decoding (opt-in) | done, but only a small gain on the model tested | below |

## 1. Weight formats

Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and BF16 are read in place from the memory-mapped GGUF file on both the CPU and Metal backends
(Q4_1 is no longer expanded to F16 at load). Mixed-format files such as Q4_K_M load as they are, and a layer may mix formats. IQ-quants
(IQ1–IQ4) are still rejected. Layouts, kernels and validation: [FORMATS.md](FORMATS.md).

* **Checked:** greedy output identical to llama.cpp on the same file for every new format (24-token and 211-token prompts), and on a real
  Llama-3.2-1B Q4_K_M file; unit tests compare every kernel path against the CPU reference on random blocks.
* **Speed (SmolLM2-135M, 16-token context):** each new format decodes within about 10–20% of Q8_0/Q4_0 and prefills at the same speed at 128 tokens,
  except BF16 prefill (≈ 9k vs 24k tok/s at 128 tokens, because it uses the exact float32 path).

## 2. Llama 3.1 / 3.2

* **RoPE scaling:** Llama 3.x ships its long-context scaling as a per-frequency table (`rope_freqs.weight` in GGUF). It is read at load and applied as
  `angle = position · θᵢ / factorᵢ` on both backends (`LlamaConfig.ropeFreqFactors`). Other scaling types (linear, YaRN) are still rejected.
* **Tokenizer:** the `llama-bpe` pre-tokenizer (digits in groups of three, case-insensitive contractions, newline handling), whole-word lookup before
  merges (tiktoken-style), and BOS added by default as llama.cpp does for this vocabulary.
* **Decode path:** a feed-forward width above 4096 (8192 in Llama 3.2 1B) cannot use the fused decode kernel; each operation now falls back on its
  own instead of the whole layer dropping to the slow path.
* **Checked:** greedy output identical to llama.cpp on Llama-3.2-1B-Instruct Q4_K_M for short prompts and a 3,499-token prompt. With the scaling table
  disabled the long-prompt output diverges, so that test does exercise the scaling. Token ids are identical to `llama-tokenize` on a 12-string battery
  (Unicode, code, digits, newlines, special tokens).
* **New command:** `alpaca tokenize <model.gguf> --prompt "…"` prints the token ids.
* **Not done:** `--chat` still wraps prompts in ChatML; Llama 3 needs its own header format.

## 3. Faster Q4_K / Q6_K decode

Dedicated single-token loops: 8 lanes per 256-element super-block, one scale decode per 32-element sub-block, vector loads, min/bias terms folded out of
the inner loop (`kq_row_dot` in `common.metal`). Llama-3.2-1B Q4_K_M decode went from about 102 to about 127 tok/s; in three interleaved rounds against
llama.cpp it measured 127–128 vs 115–134 tok/s. Q5_K, Q3_K and Q2_K still use the generic path and are untuned.

## 4. Fused dequantisation in the prefill GEMM

The tensor-op GEMM used to expand each quantised matrix to half precision in device memory and read it back (about 45% of a 128-token prefill on a 1B
K-quant model). `tensor_gemm_dq.metal` instead dequantises 64-element K chunks straight into threadgroup memory and multiplies them with an accumulating
cooperative tensor (plain, accumulating and gate+up+SiLU variants; Q4_K / Q6_K have a specialised tile fill).

It is used only where it measured faster: matrices of at least 2M weights and chunks of up to three 64-token tiles. Smaller models and longer chunks
keep the previous path. Tunables (environment variables): `ALPACA_DISABLE_FUSED_DEQUANT`, `ALPACA_FUSED_DEQUANT_MIN_ELEMS`,
`ALPACA_FUSED_DEQUANT_MAX_TILES`. A tensor-op shader library that fails to compile now prints a warning instead of silently disabling the matrix-unit path.

Llama-3.2-1B Q4_K_M prefill, tokens per second, same window:

| prompt tokens | 16 | 24 | 64 | 128 | 192 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| llama.cpp | 1083 | 937 | 2313 | 4019 | 3516 | 4275 | 4039 |
| this branch | 709 | 988 | 2636 | 3572 | 3647 | 3634 | 4459 |

## 5. Small-batch kernel (2–16 tokens)

`matmul_small.metal`: simdgroup-matrix tiles with weights dequantised into threadgroup memory and K split across the simdgroups of a threadgroup
(`mm8_*`, `mm16_*`; float32 operands, so the arithmetic matches the mat-vec kernels up to summation order). Used for short prompts and for speculative
verification. The tensor-op GEMM now starts at 17 tokens (`ALPACA_TENSOR_MIN_TOKENS`); `ALPACA_DISABLE_SMALL_BATCH=1` turns the kernel off.
On SmolLM2-135M a 16-token prefill is about 28% faster (≈ 3.5k → 4.6k tok/s).

**Known gap:** on Llama-3.2-1B Q4_K_M a 16-token prefill is still about 0.65–0.77× of llama.cpp. Several variants were tried (32-row threadgroups with a
shared activation tile, the 32×64 GEMM, the fused tensor path, half-precision operands); none gained more than about 15%, so none was kept.

## 6. Prompt-lookup speculative decoding (opt-in)

For greedy generation only. When the last five tokens have appeared before in the prompt or the output, the tokens that followed are proposed and
checked in one batched pass; the accepted prefix plus one extra token is kept, and rejected drafts back off for 16–128 tokens.

* **Use:** `alpaca generate … --speculate N` (N ≤ 7) or `GenerationConfiguration.lookupDraftTokens`. Off by default.
* **Output:** byte-identical to plain greedy decoding in every test.
* **Gain:** small on SmolLM2-135M (about +5% on code, none on prose), because a 135M-model decode step is already close to its fixed overhead and a verify
  pass costs about 1.5 decode steps. Larger models were not tested.

## Tests and how the checks were made

* `swift test --skip AlpacaIntegrationTests`: all pass (kernel tests for every format and path, tokenizer, RoPE factors). The integration tests need the
  full model set and were not run.
* Cross-checks against llama.cpp used `llama-completion … -no-cnv --temp 0` and `llama-tokenize --no-escape` on the same GGUF. Test files for the
  SmolLM formats were generated with `llama-quantize --allow-requantize` from the Q8_0 model.

## Not covered by this branch

Qwen / Gemma / Phi and other non-Llama architectures; IQ-quants; linear or YaRN RoPE scaling; Llama 3 chat template; any model other than
SmolLM2-135M and Llama-3.2-1B; iPhone/iPad; energy and thermal behaviour; a quiet-machine benchmark.
