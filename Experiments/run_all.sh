#!/bin/sh
cd "$(dirname "$0")/.."
R=Experiments/results
E=.build/release/AlpacaEval
T=Models/data/wiki.test.raw
Experiments/run_bench.sh $R/bench-main > $R/bench-main.log 2>&1
for m in 135M 360M 1.7B; do for q in f16 Q8_0 Q4_0; do
  $E Models/gguf/SmolLM2-$m-Instruct-$q.gguf $T --positions 3000 --configs default --out $R/fmt-$m-$q.json > $R/fmt-$m-$q.log 2>&1
done; done
$E Models/gguf/SmolLM2-135M-Instruct-f16.gguf $T --positions 2000 --out $R/abl-135M-f16.json > $R/abl-135M-f16.log 2>&1
$E Models/gguf/SmolLM2-135M-Instruct-Q4_0.gguf $T --positions 2000 --out $R/abl-135M-Q4_0.json > $R/abl-135M-Q4_0.log 2>&1
$E Models/gguf/SmolLM2-1.7B-Instruct-f16.gguf $T --positions 600 --out $R/abl-1.7B-f16.json > $R/abl-1.7B-f16.log 2>&1
echo finished > $R/ALL_DONE
