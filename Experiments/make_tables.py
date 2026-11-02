import json
R = "Experiments/results"; O = "paper"
S = json.load(open(f"{R}/bench-main/summary.json")); S2 = json.load(open(f"{R}/bench-repeat/summary.json"))
E = json.load(open(f"{R}/eval-summary.json")); D = json.load(open(f"{R}/derived.json"))
Q = {"f16": "F16", "Q8_0": "Q8\\_0", "Q4_0": "Q4\\_0"}
def get(rows, m, q, n): return next((r for r in rows if r["model"] == m and r["quant"] == q and r["n"] == n), None)
def f(x, d=0): return f"{x:,.{d}f}"
# speed table: prefill and decode, three lengths
L = ["\\begin{tabular}{llrrrrrr}\\toprule", " & & \\multicolumn{3}{c}{prefill, tok/s (ratio)} & \\multicolumn{3}{c}{decode, tok/s (ratio)}\\\\",
     "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}", "model & fmt & $n{=}16$ & 512 & 2048 & ctx 16 & 512 & 2048\\\\\\midrule"]
for m in ["135M", "360M", "1.7B"]:
    for q in ["f16", "Q8_0", "Q4_0"]:
        r = [get(S, m, q, n) for n in (16, 512, 2048)]
        if not r[0]: L.append(f"{m} & {Q[q]} & \\multicolumn{{6}}{{c}}{{not supported (Q6\\_K embedding)}}\\\\"); continue
        c = [f"{f(x['a_pre'])} ({x['a_pre']/x['l_pre']:.2f})" for x in r] + [f"{f(x['a_dec'])} ({x['a_dec']/x['l_dec']:.2f})" for x in r]
        L.append(f"{m} & {Q[q]} & " + " & ".join(c) + "\\\\")
    if m != "1.7B": L.append("\\midrule" if False else "")
L.append("\\bottomrule\\end{tabular}"); open(f"{O}/tab_speed.tex", "w").write("\n".join(L))
# reproducibility: ratio in two passes
L = ["\\begin{tabular}{llrrrr}\\toprule", " & & \\multicolumn{2}{c}{prefill ratio, $n{=}512$} & \\multicolumn{2}{c}{decode ratio, ctx 512}\\\\",
     "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}", "model & fmt & pass 1 & pass 2 & pass 1 & pass 2\\\\\\midrule"]
for m in ["135M", "360M", "1.7B"]:
    for q in ["f16", "Q8_0", "Q4_0"]:
        a, b = get(S, m, q, 512), get(S2, m, q, 512)
        if not a or not b: continue
        L.append(f"{m} & {Q[q]} & {a['a_pre']/a['l_pre']:.2f} & {b['a_pre']/b['l_pre']:.2f} & {a['a_dec']/a['l_dec']:.2f} & {b['a_dec']/b['l_dec']:.2f}\\\\")
L.append("\\bottomrule\\end{tabular}"); open(f"{O}/tab_repro.tex", "w").write("\n".join(L))
# quality
L = ["\\begin{tabular}{llrlr}\\toprule", "model & fmt & PPL & $\\Delta$NLL [95\\% CI] & top-1\\\\\\midrule"]
for m in ["135M", "360M", "1.7B"]:
    for q in ["f16", "Q8_0", "Q4_0"]:
        e = E["formats"].get(f"{m}-{q}")
        if not e: continue
        d = "--" if "dnll" not in e else f"{e['dnll']:+.4f} [{e['dnll_ci'][0]:+.4f}, {e['dnll_ci'][1]:+.4f}]"
        t = "--" if "top1_agree" not in e else f"{100*e['top1_agree']:.1f}\\%"
        L.append(f"{m} & {Q[q]} & {e['ppl']:.2f} & {d} & {t}\\\\")
L.append("\\bottomrule\\end{tabular}"); open(f"{O}/tab_quality.tex", "w").write("\n".join(L))
# ablation
L = ["\\begin{tabular}{llrrrr}\\toprule", "model/fmt & configuration & $\\Delta$NLL [95\\% CI] & mean KL & max KL & top-1 flips\\\\\\midrule"]
nm = {"gemm": "f16 GEMM operands", "kv16": "f16 KV cache", "default": "both (default)"}
for k, v in E["ablation"].items():
    for c in ["gemm", "kv16", "default"]:
        e = v[c]; mdl = k.replace("-", " ").replace("f16", "F16").replace("Q4_0", "Q4\\_0")
        L.append(f"{mdl} ($N{{=}}{v['ref']['n']}$) & {nm[c]} & ${e['dnll']*1e5:+.1f}$ [{e['dnll_ci'][0]*1e5:+.1f}, {e['dnll_ci'][1]*1e5:+.1f}]$\\cdot10^{{-5}}$ & {e['kl']*1e6:.2f}$\\cdot10^{{-6}}$ & {e['kl_max']*1e6:.0f}$\\cdot10^{{-6}}$ & {100*e['flips']:.2f}\\%\\\\")
    L.append("\\midrule")
L[-1] = "\\bottomrule\\end{tabular}"; open(f"{O}/tab_ablation.tex", "w").write("\n".join(L))
# efficiency
L = ["\\begin{tabular}{llrrrr}\\toprule", " & & \\multicolumn{2}{c}{prefill TFLOP/s} & \\multicolumn{2}{c}{decode GB/s}\\\\",
     "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}", "model & fmt & alpaca & llama.cpp & alpaca & llama.cpp\\\\\\midrule"]
for m in ["135M", "360M", "1.7B"]:
    for q in ["f16", "Q8_0", "Q4_0"]:
        a = next((r for r in D if r["m"] == m and r["q"] == q and r["n"] == 2048), None); b = next((r for r in D if r["m"] == m and r["q"] == q and r["n"] == 16), None)
        if a: L.append(f"{m} & {Q[q]} & {a['tflops_a']:.1f} & {a['tflops_l']:.1f} & {b['bw_a']:.0f} & {b['bw_l']:.0f}\\\\")
L.append("\\bottomrule\\end{tabular}"); open(f"{O}/tab_eff.tex", "w").write("\n".join(L))
print("tables ok")
