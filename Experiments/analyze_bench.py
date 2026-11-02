import json, sys, glob, os
D = sys.argv[1]
rows = []
for m in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["135M", "360M", "1.7B"]):
    for q in ["f16", "Q8_0", "Q4_0"]:
        if not os.path.exists(f"{D}/alpaca-{m}-{q}.json"): continue   # e.g. 1.7B Q4_0: Q6_K embedding, rejected by alpaca
        a = json.load(open(f"{D}/alpaca-{m}-{q}.json"))
        pp = {r["n_prompt"]: r["avg_ts"] for r in json.load(open(f"{D}/llamacpp-pp-{m}-{q}.json"))}
        tg = {r["n_depth"]: r["avg_ts"] for r in json.load(open(f"{D}/llamacpp-tg-{m}-{q}.json"))}
        for r in a["results"]:
            n = r["promptTokens"]
            rows.append(dict(model=m, quant=q, n=n, bytes=a["modelBytes"],
                             a_pre=r["prefillTokensPerSecond"]["median"], a_pre_sd=r["prefillTokensPerSecond"]["stdev"],
                             a_dec=r["decodeTokensPerSecond"]["median"], a_dec_sd=r["decodeTokensPerSecond"]["stdev"],
                             l_pre=pp.get(n), l_dec=tg.get(n), peak=a["peakPhysicalFootprintMiB"]))
json.dump(rows, open(f"{D}/summary.json", "w"), indent=1)
if __name__ == "__main__":
    print("model quant   n |  pre a / l  ratio |  dec a / l  ratio")
    for r in rows:
        print(f"{r['model']:>5} {r['quant']:>5} {r['n']:>4} | {r['a_pre']:8.0f} {r['l_pre'] or 0:8.0f} {r['a_pre']/(r['l_pre'] or 1):5.2f} | {r['a_dec']:6.1f} {r['l_dec'] or 0:6.1f} {r['a_dec']/(r['l_dec'] or 1):5.2f}")
