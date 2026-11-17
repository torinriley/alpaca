# Weight formats

*Author: Torin Etheridge · MIT License*

Every format below is read **in place** from the memory-mapped GGUF file (no conversion, no extra copy) on both backends.

| GGUF type | elements/block | bytes/block | notes |
|---|---|---|---|
| F16 / BF16 | 1 | 2 | BF16 runs on the exact float32 GEMM path (no half expansion: bfloat16 has a wider range than half) |
| Q8_0 | 32 | 34 | `half d; int8 q[32]` |
| Q4_0 | 32 | 18 | `half d; u8 q[16]` (low nibble = element j, high = j+16), value `d*(n-8)` |
| Q4_1 | 32 | 20 | `half d, m; u8 q[16]`, value `d*n + m` |
| Q5_0 | 32 | 22 | `half d; u32 qh; u8 q[16]`, 5th bit of element e is bit e of `qh`, value `d*(x-16)` |
| Q5_1 | 32 | 24 | `half d, m; u32 qh; u8 q[16]`, value `d*x + m` |
| Q2_K | 256 | 84 | `u8 scales[16]; u8 q[64]; half d, dmin` |
| Q3_K | 256 | 110 | `u8 hmask[32]; u8 q[64]; u8 scales[12]; half d` |
| Q4_K | 256 | 144 | `half d, dmin; u8 scales[12]; u8 q[128]` |
| Q5_K | 256 | 176 | `half d, dmin; u8 scales[12]; u8 qh[32]; u8 q[128]` |
| Q6_K | 256 | 210 | `u8 ql[128]; u8 qh[64]; i8 scales[16]; half d` |

K-quant rows must be a multiple of 256 wide. llama.cpp already stores narrower tensors (e.g. the 576-wide projections of SmolLM2-135M) in a
legacy type inside "K" files, so mixed files such as Q4_K_M load as they are; a layer may mix formats (decode then issues one fused dispatch
per run of equal formats instead of one for the whole group).

Not supported: IQ-quants (IQ1_S … IQ4_XS), which need codebook tables.

## Implementation

* CPU reference: `dequantizeBlock` in `Sources/AlpacaCore/Operations/Quantization.swift` (ports of ggml's `dequantize_row_*`).
* Metal: f16 / Q8_0 / Q4_0 keep their hand-tuned kernels. All other formats go through one primitive, `dq8<FMT>(row, unit)` in
  `common.metal`, which dequantises 8 consecutive elements (all of which share one scale / sub-block); the mat-vec, fused decode, small-batch,
  exact-GEMM, dequant-to-half and embedding kernels are each written once against it (`quant_generic.metal`, `decode_fused.metal`,
  `matmul_small.metal`).
* Small batches (2…16 tokens: short prompts, speculative verification) use `mm8_*` / `mm16_*`: simdgroup-matrix tiles with weights dequantised
  into threadgroup memory, K split across the four simdgroups of a threadgroup. Larger chunks use the tensor-op GEMM where available.

## Validation

* `KernelTests.testLinearNewFormatsAllKernelPaths`, `testEmbeddingNewFormats`, `testFastGEMMNewFormats`: every kernel family against the CPU reference, on
  random but well-formed blocks, at widths that contain whole 256-element super-blocks.
* End to end against llama.cpp (build 11429, Metal), same GGUF, greedy: identical text for SmolLM2-135M-Instruct (Q8_0) requantised with `llama-quantize --allow-requantize` to Q4_1, Q5_0,
  Q5_1, Q2_K (ffn_down), Q3_K_M, Q4_K_M, Q5_K_M, Q6_K and BF16, for a 24-token and a 211-token prompt. Between them these files contain real tensors of every
  supported type.
