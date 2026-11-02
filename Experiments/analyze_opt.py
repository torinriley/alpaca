import json, os
import sys
D = sys.argv[1] if len(sys.argv) > 1 else "Experiments/results/opt"
def alp(v, m, q):
    p = f"{D}/{v}-{m}-{q}.json"
    if not os.path.exists(p): return None
    return {r["promptTokens"]: (r["prefillTokensPerSecond"]["median"], r["decodeTokensPerSecond"]["median"]) for r in json.load(open(p))["results"]}
rows = []
print("model fmt  n | prefill tok/s: base  new  cache  llama | ratio vs llama: base  new  cache")
for m in ["135M", "360M", "1.7B"]:
    for q in ["f16", "Q8_0", "Q4_0"]:
        b, n, c = alp("base", m, q), alp("new", m, q), alp("cache", m, q)
        if not b: continue
        lp = {r["n_prompt"]: r["avg_ts"] for r in json.load(open(f"{D}/llama-pp-{m}-{q}.json"))}
        lt = {r["n_depth"]: r["avg_ts"] for r in json.load(open(f"{D}/llama-tg-{m}-{q}.json"))}
        for k in (16, 128, 512, 2048):
            l = lp[k]; rows.append(dict(m=m, q=q, n=k, base=b[k][0], new=n[k][0], cache=c[k][0] if c else None, llama=l,
                                       dbase=b[k][1], dnew=n[k][1], dllama=lt.get(k)))
            print(f"{m:>5} {q:>5} {k:>5} | {b[k][0]:7.0f} {n[k][0]:7.0f} {(c[k][0] if c else 0):7.0f} {l:7.0f} | {b[k][0]/l:5.2f} {n[k][0]/l:5.2f} {((c[k][0]/l) if c else 0):5.2f}")
json.dump(rows, open(f"{D}/summary.json", "w"), indent=1)
print("\ndecode ratio vs llama (base / new) at ctx 16, 512, 2048")
for m in ["135M", "360M", "1.7B"]:
    for q in ["f16", "Q8_0", "Q4_0"]:
        r = [x for x in rows if x["m"] == m and x["q"] == q and x["dllama"]]
        if r: print(m, q, " ".join(f"{x['n']}:{x['dbase']/x['dllama']:.2f}/{x['dnew']/x['dllama']:.2f}" for x in r))
