# Development

*Author: Torin Etheridge · 2026-10-09 · MIT License*

Requirements: Apple Silicon Mac, Xcode with Swift 6 (developed on Swift 6.4 / Xcode 27), Python 3 only for regenerating fixtures.

```sh
swift build                       # debug
swift test                        # fast suite; real-model tests skip ("need an optimised build")
swift test -c release -Xswiftc -enable-testing   # adds real-model CPU + Metal validation (needs ./Models)
Scripts/test_all.sh               # both
Scripts/fetch_models.sh [--with-reference]
```

Model location: `./Models/gguf` or `ALPACA_MODEL_DIR=/path/to/gguf-dir`. Tests that need model files skip with a message when absent.

Regenerating fixtures (only when adding tests or changing references):

```sh
python3 -m venv .venv && .venv/bin/pip install numpy torch transformers gguf sentencepiece accelerate tokenizers
Scripts/fetch_models.sh --with-reference
.venv/bin/python Scripts/generate_op_fixtures.py
.venv/bin/python Scripts/generate_tiny_llama_fixture.py
.venv/bin/python Scripts/generate_reference.py
```

Metal shaders are plain `.metal` files copied as resources and compiled at runtime. To get compiler diagnostics with line numbers, concatenate `common.metal` + the others (minus `#include "…"` lines) and run `xcrun metal -std=metal3.1 -c all.metal`.

Layout: see [ARCHITECTURE.md](ARCHITECTURE.md). iOS: `xcodebuild -scheme Alpaca -destination 'generic/platform=iOS Simulator' build`; the SwiftUI example lives in `Examples/iOS/AlpacaDemo` (a Swift package app; its path dependency refers to the repository by its checkout directory name, `Alpaca`). The macOS front end is the `alpaca` CLI.

## Architecture decision record

**ADR-1 (validation model).** SmolLM2-135M-Instruct: Apache-2.0, Llama architecture with GQA and tied embeddings, 135M parameters (fast CPU reference), ungated, GGUF in F16/Q8_0/Q4_0 available. Costs: its `gpt2`-style tokenizer means SentencePiece models are out of scope for 0.1.
**ADR-2 (shaders at runtime).** SwiftPM does not compile `.metal` files outside Xcode; runtime `makeLibrary(source:)` keeps one code path for `swift build`, Xcode and iOS at the cost of a ~20–40 ms first-load compile.
**ADR-3 (no activation quantisation).** Quantised kernels multiply dequantised weights by float32 activations. This removes one error source and keeps the CPU reference exact, but means Q8_0/Q4_0 outputs differ slightly from llama.cpp's int8-activation dot products.
**ADR-4 (f16 KV default).** Halves KV memory/bandwidth; measured logit cost documented in NUMERICAL_VALIDATION.md; `kvPrecision: .float32` available.
**ADR-5 (online-softmax attention, no standalone softmax kernel).** See ARCHITECTURE.md.

## Profiling and tuning (Phase 3 tooling)

```sh
.build/release/alpaca profile <model.gguf> --tokens 2048 --context 2200 [--prefill-batch 512] [--exact-gemm]   # per-stage GPU time, prefill
.build/release/alpaca profile <model.gguf> --tokens 1 --context-before 2048 --context 2200                    # per-stage GPU time, decode step
Scripts/compare_llamacpp.sh Benchmarks/results/<name>                                                          # full grid vs llama.cpp
```
`profile` samples GPU timestamps at encoder boundaries (needs a GPU with stage-boundary counter sampling) and reports unprofiled GPU time, CPU encode time and wall time alongside the per-stage split.

Developer environment knobs (defaults are the shipped values; used for A/B runs, not API): `ALPACA_DISABLE_TENSOR_OPS=1` (force the float32 simdgroup GEMM), `ALPACA_TENSOR_MIN_TOKENS` (batch size from which the tensor GEMM is used, 16), `ALPACA_DISABLE_FUSED_DECODE=1`, `ALPACA_DISABLE_FUSED_GATEUP=1`, `ALPACA_DISABLE_GREEDY_CHAIN=1` (synchronous decode), `ALPACA_DECODE_TARGET` / `ALPACA_DECODE_MINSPLIT` / `ALPACA_DECODE_MINCTX` (split decode attention). Benchmark on a quiet machine and compare configurations by interleaving runs.
