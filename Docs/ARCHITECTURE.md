# Architecture

*Author: Torin Etheridge · 2026-10-09 · MIT License*

## Modules

```
Alpaca            public API: LanguageModel, GenerationStream, errors        → Core, Metal, Models, Tokenizers
AlpacaCore        Tensor, CPU reference ops, KV cache, Llama CPU forward,    (Foundation only)
                  sampler, memory estimate, error metrics
AlpacaMetal       MetalContext, kernels (.metal sources), GPU Llama model    → Core
AlpacaModels      GGUF parser, Llama weight mapping, tokenizer factory       → Core, Tokenizers
AlpacaTokenizers  byte-level BPE                                             (Foundation only)
AlpacaCLI         `alpaca` executable                                        → Alpaca
```
No circular dependencies. `AlpacaCore` has no Metal or GGUF knowledge; `AlpacaMetal` has no GGUF knowledge (it receives `LlamaWeights` plus an optional mapped-region pointer).

## Tensors

`Tensor` = `TensorStorage` (64-byte aligned, zeroed, owned or borrowed) + dtype + shape + element strides + offset. Construction validates that every addressable element lies inside the storage with overflow-checked arithmetic. Views (`reshaped`, `transposed`, `slice`) share storage. Quantised dtypes must be contiguous with a last dimension divisible by 32. Operations validate dtype, shape and contiguity and throw `TensorError`.

Weights from a GGUF file are `Tensor`s whose storage *borrows* the file mapping and retains it (`keepAlive`), so a view cannot outlive the bytes.

## Transformer flow (per layer, identical on CPU and GPU)

```
h = rmsnorm(x, attn_norm);  q,k,v = h·Wq, h·Wk, h·Wv
q,k = rope(q,k, position)   (interleaved pairs: the layout of GGUF Llama weights)
cache[layer][pos] = k,v;    ctx = causal_gqa_attention(q, cache[0...pos])
x  += ctx·Wo
h = rmsnorm(x, ffn_norm);   x += (silu(h·Wgate) ⊙ (h·Wup))·Wdown
```
Final: `rmsnorm(x_last, output_norm)·W_out` (tied to the embedding when `output.weight` is absent). Prefill and decode are the same code with `tokens.count > 1` or `== 1`; only the last position's logits are computed.

## KV cache

Per layer `[capacity, kvHeads, headDim]` for K and V, contiguous, row = absolute position. Written in place (bounds- and position-checked), read in place by attention bounded by `length`; reset only zeroes the length. GQA needs no special layout: query head `h` reads KV head `h / (heads / kvHeads)`. CPU: float32. Metal: float16 by default (`KVPrecision.float16`) or float32, with 32 zeroed padding rows so tiled attention can read whole blocks. Each `generate` call creates its own cache (`MetalLlamaSession` / `KVCache`), so there is no shared mutable state between generations; a session also refuses re-entrant `forward`.

## Metal backend

- `MetalContext` compiles all `Kernels/*.metal` at first use with `makeLibrary(source:)` (works under `swift build`, Xcode and on iOS without a build plugin) and caches pipelines.
- **Weights**: the whole mmap'd GGUF file is wrapped as one no-copy `MTLBuffer`; each weight is `(buffer, offset)`. Resident weight memory ≈ file size and nothing is duplicated (unified memory). Converted tensors (Q4_1 → F16) get their own buffers.
- **Dispatch**: one command buffer and one serial compute encoder per `forward` call (all layers, all prefill chunks); chunks of up to 512 tokens. A single-token step is 6 dispatches per layer (fused: norm+Q/K/V, RoPE+KV store, attention (+merge for long contexts), out-projection with residual, norm+gate+up+SiLU, down-projection with residual); a prefill chunk is ~7 plus a weight-expansion dispatch per quantised matrix. **Greedy decoding runs on the GPU**: an arg-max kernel feeds the next step's embedding directly and three command buffers are kept in flight, so the CPU only streams tokens out; other samplers read the logits (a no-copy view of the shared buffer) and take one CPU round trip per token.
- **Precision modes**: `GEMMPrecision.fast` (default) runs prefill projections on the GPU's matrix hardware (Metal 4 tensor ops; half operands, float32 accumulate) on Apple10-family GPUs and otherwise falls back to `.exact` (float32 simdgroup-matrix GEMM). `KVPrecision.float16` (default) or `.float32`. Details and measured effects: Docs/NUMERICAL_VALIDATION.md.
- **Sessions**: each session owns its KV cache, scratch, command queue and greedy-chain buffers; finished sessions are recycled by `LanguageModel` (one idle session retained; `trimMemory()` releases it).
- **Kernels** (`Sources/AlpacaMetal/Kernels`):
  | file | kernels | notes |
  |---|---|---|
  | `decode_fused.metal` | `dec_{f16,q8_0,q4_0}`, `rope_qkv_store_kv{16,32}` | single-token path: RMSNorm folded into the mat-vec prologue (input row staged in threadgroup memory), Q/K/V in one dispatch, gate+up+SiLU in one, residual accumulate; RoPE of q and RoPE+cache append of k/v in one dispatch (also used for prefill) |
  | `matvec.metal` | `mv_{f16,q8_0,q4_0}_tb{1,8}` | small batches (< 16 tokens): one simdgroup per output row, `TB` tokens share each weight read; optional accumulate into the output |
  | `matmul.metal` | `mm_{f16,q8_0,q4_0}` | `.exact` batched path: 32-token × 64-row tiles, 8×8 float32 `simdgroup_matrix`, optional accumulate |
  | `tensor_gemm.metal` | `gemm_tensor_half_w[_acc]`, `gemm_tensor_gateup_silu`, `dequant_{q8_0,q4_0}_to_half` | `.fast` batched path (own library, Metal 4.0, `MetalPerformancePrimitives matmul2d`): 64×64 tiles, half weights; quantised matrices are expanded one at a time into a reusable scratch; `_acc` accumulates into the residual; gateup computes both products in registers and applies SiLU·mul before storing |
  | `attention_prefill.metal` | `attn_prefill_hd{64,128}_kv16` | FlashAttention-style: 64-query tile per threadgroup, K/V blocks of 32 staged once in threadgroup memory and shared by 4 simdgroups, two 8×8 row tiles per simdgroup, causal block skipping, online softmax with two lanes per row, output rescaling by a diagonal-matrix MMA |
  | `attention_decode.metal` | `attn_decode_split_hd{64,128}`, `attn_decode_merge` | split-K decode attention: one threadgroup per (KV head, position split) handles all query heads of the GQA group, quad-per-position layout; ~8 splits merged by a second kernel |
  | `attention.metal` | `attention_kv{16,32}`, `rope_f32` | general fallback (any head dim up to 256, f32 KV, short batches) |
  | `norm_embed.metal` | `rmsnorm_f32`, `embed_*`, `argmax_logits` | float32 sum of squares; GPU greedy sampling |
  | `elementwise.metal` | `add`, `mul`, `silu`, `silu_mul` | non-fused fallbacks |
- Softmax is fused into the attention kernels (online form) and validated through them; there is deliberately no standalone softmax kernel.
- All accumulation is float32. Weights are read at their stored precision (f16 or quantised).

## Quantisation design

Q8_0 (34-byte blocks of 32: `half d`, `int8 q[32]`) and Q4_0 (18-byte blocks: `half d`, 16 nibble bytes; element j low nibble, j+16 high nibble, value `d·(nibble−8)`) are consumed directly by the mat-vec/GEMM kernels. Activations stay float32. The reference quantiser in `AlpacaCore` reproduces `gguf`-package output byte for byte.

## Memory lifecycle

`LanguageModel.load`: mmap → parse/validate → map tensors (zero-copy views) → **estimate and check budget** → create GPU model (wrap file as buffer) → ready. `generate`: allocate session (KV + scratch) → run on a dedicated thread → release on completion/cancellation. `unload()` cancels running generations and drops weight references. See `MemoryEstimate`: total ≈ weights (≈ file size) + KV (`2·layers·context·kvWidth·bytes`) + scratch (`4·prefillBatch·Σwidths + 4·vocab`) + 96 MiB runtime allowance.
