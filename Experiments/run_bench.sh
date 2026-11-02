#!/bin/sh
# Multi-model benchmark: alpaca vs llama.cpp on SmolLM2 135M / 360M / 1.7B in F16, Q8_0 and Q4_0 (same GGUF files).
# Usage: Experiments/run_bench.sh <out-dir> [models...]
set -u
cd "$(dirname "$0")/.."
OUT=$1; shift
MODELS=${*:-"135M 360M 1.7B"}
mkdir -p "$OUT"
{ date; uptime; swift --version 2>&1 | head -1; sw_vers; sysctl -n machdep.cpu.brand_string hw.memsize; llama-bench --version 2>&1 | head -3; pmset -g therm 2>&1; } > "$OUT/environment.txt" 2>&1
PL=16,128,512,2048; CS=16,512,2048
for m in $MODELS; do for q in f16 Q8_0 Q4_0; do
  F="Models/gguf/SmolLM2-$m-Instruct-$q.gguf"
  echo "== $m $q $(uptime | sed 's/.*load/load/')"
  .build/release/alpaca bench "$F" --backend metal --runs 7 --prompt-lengths $PL --context-sweep $CS --context 2300 --json "$OUT/alpaca-$m-$q.json" > "$OUT/alpaca-$m-$q.txt" 2>&1
  llama-bench -m "$F" -ngl 99 -r 7 -p $PL -n 0 -o json > "$OUT/llamacpp-pp-$m-$q.json" 2>/dev/null
  llama-bench -m "$F" -ngl 99 -r 7 -p 0 -n 64 -d $CS -o json > "$OUT/llamacpp-tg-$m-$q.json" 2>/dev/null
done; done
echo finished > "$OUT/DONE"
