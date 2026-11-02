"""Figures for the GEMM scaling study, from Experiments/results/opt*/summary.json."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "Experiments/results/"
OUT = "Docs/images/"
GREY, BLUE, ORANGE = "#8c8c8c", "#2a6fdb", "#e07b00"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

def load(d):
    return json.load(open(R + d + "/summary.json"))

def ratios(rows, m, q, ns):
    out = {"base": [], "new": [], "cache": []}
    for n in ns:
        r = next(x for x in rows if x["m"] == m and x["q"] == q and x["n"] == n)
        out["base"].append(r["base"] / r["llama"])
        out["new"].append(r["new"] / r["llama"])
        out["cache"].append(r["cache"] / r["llama"] if r["cache"] else np.nan)
    return out

def bars(ax, labels, d, title):
    x = np.arange(len(labels)); w = 0.26
    for i, (k, c, name) in enumerate([("base", GREY, "before"), ("new", BLUE, "after"), ("cache", ORANGE, "after + --prefill-cache")]):
        v = np.array(d[k])
        if np.isnan(v).all(): continue
        b = ax.bar(x + (i - 1) * w, v, w, color=c, label=name)
        for xi, vi in zip(x + (i - 1) * w, v):
            if not np.isnan(vi): ax.text(xi, vi + 0.02, f"{vi:.2f}", ha="center", fontsize=8)
    ax.axhline(1, color="black", lw=1, ls="--")
    ax.set_xticks(x, labels); ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.set_ylabel("prefill speed vs llama.cpp (x)")

# 1.7B, run B (opt-1.7B-r1) - quietest of the three runs
rows = load("opt-1.7B-r1")
ns = [128, 512, 2048]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
bars(axs[0], [f"{n} tokens" for n in ns], ratios(rows, "1.7B", "f16", ns), "SmolLM2-1.7B  F16")
bars(axs[1], [f"{n} tokens" for n in ns], ratios(rows, "1.7B", "Q8_0", ns), "SmolLM2-1.7B  Q8_0")
axs[0].legend(frameon=False, loc="upper left", fontsize=8); axs[1].legend(frameon=False, loc="upper left", fontsize=8)
fig.suptitle("Prefill speed relative to llama.cpp (above the dashed line = alpaca faster)", fontsize=10, y=0.99)
fig.tight_layout(); fig.savefig(OUT + "scaling-1.7B.png", dpi=160); plt.close(fig)

# small models at 512 tokens
rows = load("opt")
labs, d = [], {"base": [], "new": [], "cache": []}
for m in ["135M", "360M"]:
    for q, ql in [("f16", "F16"), ("Q8_0", "Q8_0"), ("Q4_0", "Q4_0")]:
        r = ratios(rows, m, q, [512])
        labs.append(f"{m}\n{ql}")
        for k in d: d[k].append(r[k][0])
fig, ax = plt.subplots(figsize=(10, 3.6))
bars(ax, labs, d, "135M and 360M at 512 tokens: unchanged without the cache, faster with it")
ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=3); ax.set_ylim(0, 1.9)
fig.tight_layout(); fig.savefig(OUT + "scaling-small.png", dpi=160); plt.close(fig)
print("done")
