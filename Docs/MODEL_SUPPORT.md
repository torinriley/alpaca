# Model support

*Author: Torin Etheridge · 2026-10-09 · MIT License*

## Reference configuration (validated)

**SmolLM2-135M-Instruct** (HuggingFaceTB, Apache-2.0), GGUF conversions by bartowski. `general.architecture = llama`.

| parameter | value |
|---|---|
| layers / hidden / FFN | 30 / 576 / 1536 |
| heads / KV heads / head dim | 9 / 3 / 64 (grouped-query, group size 3) |
| vocabulary / trained context | 49,152 / 8,192 |
| RoPE | θ = 100000, full head dimension, no scaling, interleaved-pair layout in GGUF |
| RMSNorm ε | 1e-5 |
| output | tied to token embedding (no `output.weight`) |
| tokenizer | `gpt2` byte-level BPE, pre-tokenizer `smollm` (isolate each Unicode numeric char, then GPT-2 regex), BOS not added, EOS = `<|im_end|>` (id 2) |
| chat format | ChatML (`<|im_start|>role\n…<|im_end|>\n`) — applied by the caller |

Files validated: `SmolLM2-135M-Instruct-{f16,Q8_0,Q4_0}.gguf`. The Q4_0 file mixes types: 207 Q4_0, 3 Q4_1 (`ffn_down` of blocks 0, 1, 10; expanded to F16 at load), 1 Q8_0 (`token_embd`), 61 F32 norms.

## Contract for other models

Accepted: `general.architecture == "llama"`; head count, KV head count, embedding/FFN/block/context lengths present; head dim = `attention.key_length` or hidden/heads, even, multiple of 32 and ≤ 256 on Metal; `rope.dimension_count` (if present) equal to head dim; no `rope.scaling.type` (other than `none`); no experts; norm tensors F32; projection/embedding tensors F32 (CPU only), F16, Q8_0, Q4_0 (or Q4_1, expanded); projection input width a multiple of 32 (quantised) / 8 (F16).
Rejected with an explicit error (never silently reinterpreted): other architectures, RoPE scaling (e.g. Llama 3.1), K-quants, Q5_x, BF16, MoE, SentencePiece/Unigram tokenizers, tokenizers with other pre-tokenizers (e.g. Llama-3), big-endian files, GGUF v1.

Other models of the same shape (e.g. Llama-3.2-1B uses RoPE scaling and a different pre-tokenizer → *not* supported yet) have **not** been tested.

## Memory model (SmolLM2-135M)

| item | F16 | Q8_0 | Q4_0 |
|---|---|---|---|
| file = mapped weights | 271 MB | 145 MB | 92 MB (+5 MB converted Q4_1→F16 copy) |
| KV cache @ 4096 ctx, f16 / f32 | 90 / 181 MiB | same | same |
| scratch (prefill batch 128) | ~3 MiB | same | same |
| runtime allowance | 96 MiB | same | same |

Weights are file-backed pages of the mapping shared with the GPU buffer, not a second copy. KV and scratch are allocated per generation session.
