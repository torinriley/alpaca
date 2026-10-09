# Benchmarks

*Author: Torin Etheridge · 2026-10-09 · MIT License*

Raw measurements behind [Docs/PERFORMANCE.md](../Docs/PERFORMANCE.md). Every JSON file records hardware, OS, model, backend, per-cell median/min/max/standard deviation, load times, peak memory and thermal state.

| directory | contents |
|---|---|
| `results/phase2-initial/` | First complete implementation: Metal backend before the Phase 3 optimisation work, plus a CPU-backend run and the first llama.cpp comparison |
| `results/phase3-baseline/` | Phase 3 baseline, re-measured under the conditions of the final run (alpaca and llama.cpp back to back) |
| `results/phase3-final/` | After the Phase 3 optimisations; the numbers quoted in the README |

Reproduce a full grid (alpaca and llama.cpp, three formats, 16…4096 tokens, 7 repetitions):

```sh
Scripts/fetch_models.sh
Scripts/compare_llamacpp.sh Benchmarks/results/<name>
```

Caveat recorded in the PERFORMANCE document: the Phase 3 measurements were taken on a machine with background load (load average 5–7). Compare alpaca and llama.cpp only within one run; rerun on an idle machine for publishable absolute numbers.
