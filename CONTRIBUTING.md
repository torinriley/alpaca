# Contributing

*Author: Torin Etheridge · 2026-10-09 · MIT License*

- Correctness first: every operation needs a CPU reference, an independently generated test vector
  (`Scripts/generate_*.py`, never alpaca's own output) and a stated tolerance with a reason.
- Do not loosen a tolerance to make a test pass; investigate and document the cause (see the f16 KV cache analysis in
  `Docs/NUMERICAL_VALIDATION.md` for the expected standard).
- Every optimisation needs a baseline, a correctness check and a benchmark in the PR description.
- No new dependencies without a written justification. No external inference runtimes.
- Run `Scripts/test_all.sh` before sending changes; state which hardware you ran on.
