import json, glob, re, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})
R = "Experiments/results"; O = "paper/figs"
C = {"135M": "#0072B2", "360M": "#E69F00", "1.7B": "#CC3311"}; MK = {"f16": "o", "Q8_0": "s", "Q4_0": "^"}
P = {"135M": 134.5e6, "360M": 361.8e6, "1.7B": 1.71e9}
# matmul parameters per model (excluding the embedding table) for the FLOP estimate
MM = {"135M": 30 * (2 * 576**2 + 2 * 576 * 192 + 3 * 576 * 1536), "360M": 32 * (2 * 960**2 + 2 * 960 * 320 + 3 * 960 * 2560),
      "1.7B": 24 * (4 * 2048**2 + 3 * 2048 * 8192)}
S = json.load(open(f"{R}/bench-main/summary.json")); S2 = json.load(open(f"{R}/bench-repeat/summary.json"))
def sel(rows, m, q, key): return [(r["n"], r[key]) for r in rows if r["model"] == m and r["quant"] == q and r[key]]
# Fig 1: ratio vs llama.cpp
fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.5))
for m in C:
    for q in MK:
        for a, (ka, kl, ttl) in zip(ax, [("a_pre", "l_pre", "prefill"), ("a_dec", "l_dec", "decode")]):
            rows = [(r["n"], r[ka] / r[kl]) for r in S if r["model"] == m and r["quant"] == q and r[kl]]
            if rows: a.plot(*zip(*rows), marker=MK[q], color=C[m], lw=1, ms=4, label=f"{m} {q}" if ttl == "prefill" else None, ls={"f16": "-", "Q8_0": "--", "Q4_0": ":"}[q])
for a, t in zip(ax, ["Prefill (prompt of $n$ tokens)", "Decode (after $n$ tokens of context)"]):
    a.axhline(1, color="k", lw=.8); a.set_xscale("log", base=2); a.set_xlabel("$n$ (tokens)"); a.set_title(t, fontsize=8)
    a.set_ylabel("alpaca / llama.cpp throughput")
ax[0].legend(ncol=1, fontsize=5.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0)) if False else None
h = [plt.Line2D([], [], color=C[m], lw=2, label=m) for m in C] + [plt.Line2D([], [], color="k", marker=MK[q], ls="", ms=4, label=q) for q in MK]
ax[1].legend(handles=h, fontsize=6, frameon=False, ncol=2, loc="lower left")
fig.tight_layout(); fig.savefig(f"{O}/ratio.pdf"); plt.close()
# Fig 2: quality vs size
E = json.load(open(f"{R}/eval-summary.json"))
fig, ax = plt.subplots(figsize=(3.3, 2.5))
for m in C:
    pts = [(E["formats"][f"{m}-{q}"]["size"] / 2**30, E["formats"][f"{m}-{q}"].get("dnll", 0), E["formats"][f"{m}-{q}"].get("dnll_ci")) for q in MK if f"{m}-{q}" in E["formats"]]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C[m], lw=1, label=m)
    for (x, y, ci), q in zip(pts, [q for q in MK if f"{m}-{q}" in E["formats"]]):
        ax.errorbar(x, y, yerr=None if ci is None else [[y - ci[0]], [ci[1] - y]], fmt=MK[q], color=C[m], ms=4, capsize=2, lw=1)
ax.set_xscale("log"); ax.set_xlabel("file size (GiB)"); ax.set_ylabel(r"$\Delta$NLL vs F16 (nats/token)"); ax.legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig(f"{O}/quality.pdf"); plt.close()
# Fig 3: ablation KL
fig, ax = plt.subplots(figsize=(3.3, 2.4))
keys = [k for k in ["135M-f16", "135M-Q4_0", "360M-f16", "1.7B-f16"] if k in E["ablation"]]
w = .26
for j, (c, lab) in enumerate([("gemm", "f16 GEMM operands"), ("kv16", "f16 KV cache"), ("default", "both (default)")]):
    v = [E["ablation"][k][c]["kl"] for k in keys]; lo = [E["ablation"][k][c]["kl_ci"][0] for k in keys]; hi = [E["ablation"][k][c]["kl_ci"][1] for k in keys]
    ax.bar(np.arange(len(keys)) + (j - 1) * w, v, w, yerr=[np.array(v) - lo, np.array(hi) - v], label=lab, capsize=1.5, color=["#88CCEE", "#DDCC77", "#332288"][j])
ax.set_yscale("log"); ax.set_xticks(range(len(keys))); ax.set_xticklabels([k.replace("-", "\n") for k in keys])
ax.set_ylabel(r"mean KL$(p_{\mathrm{exact}}\,\|\,p)$ (nats)"); ax.legend(frameon=False, fontsize=6, loc="upper left"); ax.set_ylim(1e-7, 1e-4)
fig.tight_layout(); fig.savefig(f"{O}/kl.pdf"); plt.close()
# Fig 4: sustained
fs = sorted(glob.glob(f"{R}/extra/sustained-*.json"), key=lambda f: int(re.findall(r"-(\d+)\.json", f)[0]))[:-1]
v = [json.load(open(f))["results"][0]["decodeTokensPerSecond"]["median"] for f in fs]
T = [int(l.split()[1]) for l in open(f"{R}/extra/sustained.log")][:len(v)]; T = (np.array(T) - T[0]) / 60
fig, ax = plt.subplots(figsize=(3.3, 2.0)); ax.plot(T, v, lw=.8, color=C["360M"]); ax.set_ylim(0, 300)
ax.set_xlabel("minutes"); ax.set_ylabel("decode tok/s"); fig.tight_layout(); fig.savefig(f"{O}/sustained.pdf"); plt.close()
# Fig 5: prefill stage breakdown
def stages(path):
    d = {}
    for l in open(path):
        m = re.match(r"\s+(.+?)\s{2,}([\d.]+) ms\s+([\d.]+)%", l)
        if m: d[m.group(1).strip()] = float(m.group(3))
    return d
grp = {"FFN gate+up+SiLU": ["ffn gate+up+silu"], "FFN down": ["ffn down (+residual)"], "QKV": ["qkv projections"], "attention": ["attention"], "out proj": ["attn out projection (+residual)"]}
fig, ax = plt.subplots(figsize=(3.3, 1.9)); left = {}
for i, m in enumerate(["135M", "1.7B"]):
    st = stages(f"{R}/extra/profile-prefill2048-{m}-Q8_0.txt"); x = 0
    used = 0
    for j, (g, ks) in enumerate(grp.items()):
        val = sum(st.get(k, 0) for k in ks); used += val
        ax.barh(i, val, left=x, color=plt.cm.tab10(j), label=g if i == 0 else None); x += val
    ax.barh(i, 100 - used, left=x, color="#bbbbbb", label="other" if i == 0 else None)
ax.set_yticks([0, 1]); ax.set_yticklabels(["135M", "1.7B"]); ax.set_xlabel("share of GPU time, Q8_0 prefill of 2048 tokens (%)")
ax.legend(fontsize=5.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.38)); fig.tight_layout(); fig.savefig(f"{O}/stages.pdf"); plt.close()
# derived tables
rows = []
for m in C:
    for q in MK:
        for r in S:
            if r["model"] == m and r["quant"] == q and r["n"] in (16, 512, 2048):
                rows.append(dict(m=m, q=q, n=r["n"], a_pre=r["a_pre"], l_pre=r["l_pre"], a_dec=r["a_dec"], l_dec=r["l_dec"], bytes=r["bytes"],
                                 tflops_a=2 * MM[m] * r["a_pre"] / 1e12, tflops_l=2 * MM[m] * (r["l_pre"] or 0) / 1e12,
                                 bw_a=r["bytes"] * r["a_dec"] / 1e9, bw_l=r["bytes"] * (r["l_dec"] or 0) / 1e9))
json.dump(rows, open(f"{R}/derived.json", "w"), indent=1)
print("figs ok", len(rows))
