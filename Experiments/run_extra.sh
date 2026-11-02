#!/bin/sh
# Experiments that need an otherwise idle GPU: per-stage profiles, run-to-run drift, sustained-load decode.
cd "$(dirname "$0")/.."
R=Experiments/results/extra; mkdir -p $R
A=.build/release/alpaca
for m in 135M 1.7B; do
  F=Models/gguf/SmolLM2-$m-Instruct-Q8_0.gguf
  $A profile $F --tokens 2048 --context 2300 > $R/profile-prefill2048-$m-Q8_0.txt 2>&1
  $A profile $F --tokens 1 --context-before 2048 --context 2300 > $R/profile-decode2048-$m-Q8_0.txt 2>&1
  $A profile $F --tokens 1 --context-before 16 --context 2300 > $R/profile-decode16-$m-Q8_0.txt 2>&1
done
# second independent pass of the smallest grid cells to estimate drift between runs (same files, same flags)
Experiments/run_bench.sh Experiments/results/bench-repeat 135M 360M > $R/bench-repeat.log 2>&1
# sustained decode: ~10 minutes of back-to-back 512-token generations
F=Models/gguf/SmolLM2-360M-Instruct-Q8_0.gguf
end=$(( $(date +%s) + 600 )); i=0
while [ $(date +%s) -lt $end ]; do
  $A bench $F --backend metal --runs 1 --prompt-lengths 16 --context-sweep 16 --decode 512 --json $R/sustained-$i.json > /dev/null 2>&1
  echo "$i $(date +%s) $(pmset -g therm 2>&1 | grep -i -E 'speed_limit|cpu_scheduler' | tr '\n' ' ')" >> $R/sustained.log
  i=$((i+1))
done
echo done > $R/DONE
