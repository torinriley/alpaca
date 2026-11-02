import json, glob, os, sys
import numpy as np
R = sys.argv[1] if len(sys.argv) > 1 else "Experiments/results"
rng = np.random.default_rng(0)
def boot(x, f=np.mean, n=2000):
    x = np.asarray(x); idx = rng.integers(0, len(x), (n, len(x)))
    return np.percentile(f(x[idx], axis=1), [2.5, 97.5])
out = {"formats": {}, "ablation": {}}
# --- quantisation: per-position NLL of the default configuration, paired against F16 of the same model size
for m in ["135M", "360M", "1.7B"]:
    base = None
    for q in ["f16", "Q8_0", "Q4_0"]:
        p = f"{R}/fmt-{m}-{q}.json"
        if not os.path.exists(p): continue
        d = json.load(open(p)); c = d["default"]
        nll = np.array(c["nll"]); top = np.array(c["top1"])
        if q == "f16": base = (nll, top)
        e = dict(ppl=float(np.exp(nll.mean())), ppl_ci=[float(np.exp(v)) for v in boot(nll)], n=len(nll), size=os.path.getsize(f"Models/gguf/SmolLM2-{m}-Instruct-{q}.gguf"))
        if base is not None and q != "f16":
            dn = nll - base[0]
            e["dnll"] = float(dn.mean()); e["dnll_ci"] = [float(v) for v in boot(dn)]
            e["top1_agree"] = float((top == base[1]).mean())
        out["formats"][f"{m}-{q}"] = e
# --- precision ablation: shipped shortcuts vs the most precise configuration, same positions
for p in sorted(glob.glob(f"{R}/abl-*.json")):
    key = os.path.basename(p)[4:-5]; d = json.load(open(p)); ref = d["ref"]; e = {}
    for c in ["gemm", "kv16", "default"]:
        nll = np.array(d[c]["nll"]); dn = nll - np.array(ref["nll"]); kl = np.array(d[c]["kl"])
        e[c] = dict(ppl=float(np.exp(nll.mean())), dnll=float(dn.mean()), dnll_ci=[float(v) for v in boot(dn)],
                    kl=float(kl.mean()), kl_ci=[float(v) for v in boot(kl)], kl_max=float(kl.max()),
                    flips=float((np.array(d[c]["top1"]) != np.array(ref["top1"])).mean()),
                    ms=d[c]["prefillMsPerWindow"])
    e["ref"] = dict(ppl=float(np.exp(np.mean(ref["nll"]))), n=len(ref["nll"]), ms=ref["prefillMsPerWindow"])
    out["ablation"][key] = e
json.dump(out, open(f"{R}/eval-summary.json", "w"), indent=1)
for k, v in out["formats"].items(): print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items() if a != "ppl_ci"})
for k, v in out["ablation"].items():
    print(k, "ref ppl %.3f n=%d" % (v["ref"]["ppl"], v["ref"]["n"]))
    for c in ["gemm", "kv16", "default"]: print("  ", c, {a: (float("%.3g" % b) if isinstance(b, float) else b) for a, b in v[c].items()})
