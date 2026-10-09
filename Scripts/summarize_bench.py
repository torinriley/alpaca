#!/usr/bin/env python3
# Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
# Author: Torin Etheridge · 2026-10-09
"""Prints a markdown table of alpaca vs llama.cpp medians from a results directory."""
import json, sys, glob, os
d = sys.argv[1]
def alp(q):
    p = f"{d}/alpaca-{q}.json"
    if not os.path.exists(p): return None
    j = json.load(open(p)); pre, dec = {}, {}
    for r in j["results"]:
        pre[r["promptTokens"]] = r["prefillTokensPerSecond"]["median"]; dec[r["promptTokens"]] = r["decodeTokensPerSecond"]["median"]
    return pre, dec, j
def lc(kind, q):
    p = f"{d}/llamacpp-{kind}-{q}.json"
    if not os.path.exists(p): return {}
    return {(r["n_prompt"] if kind == "pp" else r["n_depth"]): r["avg_ts"] for r in json.load(open(p))}
for q in ["f16", "Q8_0", "Q4_0"]:
    a = alp(q)
    if not a: continue
    pre, dec, j = a; lpp, ltg = lc("pp", q), lc("tg", q)
    print(f"\n### {q}  (peak resident {j.get('peakResidentMiB')} MiB)\n")
    print("| tokens / context | prefill alpaca | prefill llama.cpp | ratio | decode alpaca | decode llama.cpp | ratio |\n|---|---|---|---|---|---|---|")
    for n in sorted(pre):
        ap, ad = pre[n], dec.get(n, 0); lp, ld = lpp.get(n), ltg.get(n)
        f = lambda x: f"{x:,.0f}" if x else "–"
        r = lambda a, b: f"{a/b:.2f}×" if a and b else "–"
        print(f"| {n} | {f(ap)} | {f(lp)} | {r(ap, lp)} | {f(ad)} | {f(ld)} | {r(ad, ld)} |")
