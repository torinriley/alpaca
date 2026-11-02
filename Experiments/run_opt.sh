#!/bin/sh
# Interleaved comparison: baseline build vs optimised build (+prefill cache for quantised) vs llama.cpp, same files, same time window.
cd "$(dirname "$0")/.."
O=${OUT:-Experiments/results/opt}; mkdir -p $O
BASE=${BASE:-../base/alpaca}; NEW=.build/release/alpaca
PL=16,128,512,2048; CS=16,512,2048
{ date; uptime; sysctl -n machdep.cpu.brand_string; git rev-parse --short HEAD; git status -s | head -20; } > $O/environment.txt 2>&1
for m in ${MODELS:-135M 360M 1.7B}; do for q in f16 Q8_0 Q4_0; do
  F=Models/gguf/SmolLM2-$m-Instruct-$q.gguf
  [ $m = 1.7B ] && [ $q = Q4_0 ] && continue
  echo "== $m $q"
  $BASE bench $F --backend metal --runs 5 --prompt-lengths $PL --context-sweep 16 --context 2300 --json $O/base-$m-$q.json > /dev/null 2>&1
  $NEW  bench $F --backend metal --runs 5 --prompt-lengths $PL --context-sweep 16 --context 2300 --json $O/new-$m-$q.json > /dev/null 2>&1
  if [ $q != f16 ]; then $NEW bench $F --backend metal --runs 5 --prompt-lengths $PL --context-sweep 16 --context 2300 --prefill-cache --json $O/cache-$m-$q.json > /dev/null 2>&1; fi
  llama-bench -m $F -ngl 99 -r 5 -p $PL -n 0 -o json > $O/llama-pp-$m-$q.json 2>/dev/null
  llama-bench -m $F -ngl 99 -r 5 -p 0 -n 64 -d $CS -o json > $O/llama-tg-$m-$q.json 2>/dev/null
done; done
echo done > $O/DONE
