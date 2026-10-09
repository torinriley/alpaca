#!/bin/sh
# Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
# Author: Torin Etheridge · 2026-10-09
# Runs alpaca and llama.cpp over the same grid on the same GGUF files and writes JSON into $1 (default Benchmarks/results/current).
# Grid: prefill 16,128,512,1024,2048,4096 ; decode 64 tokens after context 16,128,512,1024,2048,4096. 7 repetitions each.
set -eu
cd "$(dirname "$0")/.."
OUT=${1:-Benchmarks/results/current}
QUANTS=${QUANTS:-"f16 Q8_0 Q4_0"}
mkdir -p "$OUT"
swift build -c release 2>&1 | tail -1
{ swift --version 2>&1 | head -1; sw_vers; sysctl -n machdep.cpu.brand_string; llama-bench --version 2>&1 | grep -E "version|build" | head -2; } > "$OUT/environment.txt" 2>&1 || true
for q in $QUANTS; do
  F="Models/gguf/SmolLM2-135M-Instruct-$q.gguf"
  .build/release/alpaca bench "$F" --backend metal --runs 7 --prompt-lengths 16,128,512,1024,2048,4096 \
      --context-sweep 16,128,512,1024,2048,4096 --context 4400 --json "$OUT/alpaca-$q.json" > "$OUT/alpaca-$q.txt"
  if command -v llama-bench >/dev/null; then
    llama-bench -m "$F" -ngl 99 -r 7 -p 16,128,512,1024,2048,4096 -n 0 -o json > "$OUT/llamacpp-pp-$q.json" 2>/dev/null
    llama-bench -m "$F" -ngl 99 -r 7 -p 0 -n 64 -d 16,128,512,1024,2048,4096 -o json > "$OUT/llamacpp-tg-$q.json" 2>/dev/null
  fi
done
python3 Scripts/summarize_bench.py "$OUT"
