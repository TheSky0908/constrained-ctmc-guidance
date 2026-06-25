#!/usr/bin/env python3
"""Leaderboard for all steps=1024 toxic-prefix runs (CBG + adual), N=1000, τ=0.5.

For each run reports viol@0.5, PPL, mean_tox, AND fluency proxies computed from
the continuations: repetition-collapse rate (distinct-1 < 0.4) and mean distinct-1.
Ranks by viol; also flags the 'fluency-clean best' = lowest viol among configs whose
collapse rate is not elevated (≤ FLOOR_COLLAPSE + 3 pp), since low viol is often
bought by degeneration (PPL alone is a misleading quality proxy).
Writes results/leaderboard_steps1024.md.
"""
import csv, glob, os, re, json, statistics

ROOT = "/local/scratch/zhiheng/guidance"
OUT = os.path.join(ROOT, "outputs/owt/mdlm_owt")
RES = os.path.join(ROOT, "experiments/text_toxicity/results")
LB = os.path.join(RES, "leaderboard_steps1024.md")

CBG = re.compile(r"tox_cbg_gamma([\d.]+)_le0\.50_toxpfx_n1000_steps1024_results\.csv$")
AD = re.compile(r"tox_adual_C([-\d.]+)_rho([\d.]+)_l0([\d.]+)_lmax([\d.]+)_le0\.50_toxpfx_n1000_steps1024_results\.csv$")
CMAP = {"0.01005": "0.99", "0.2231": "0.8", "0.6931": "0.5", "1.204": "0.3", "1.2040": "0.3"}


def fluency(samples_path):
    try:
        recs = json.load(open(samples_path))["records"]
    except Exception:
        return None, None
    d1, col, n = [], 0, 0
    for r in recs:
        w = r.get("continuation", "").split()
        if not w:
            continue
        n += 1
        ratio = len(set(w)) / len(w)
        d1.append(ratio)
        if ratio < 0.4:
            col += 1
    if not n:
        return None, None
    return statistics.mean(d1), 100 * col / n


def read(path):
    try:
        r = list(csv.DictReader(open(path)))[-1]
        viol = float(r["viol_tau_0.5"]) * 100
        ppl = float(r["ppl_gpt2xl"])
        mt = float(r["mean_toxicity"])
    except Exception:
        return None
    d1, col = fluency(path.replace("_results.csv", "_samples.json"))
    return dict(viol=viol, ppl=ppl, mt=mt, d1=d1, col=col)


def main():
    rows = []
    for f in glob.glob(os.path.join(OUT, "*_toxpfx_n1000_steps1024_results.csv")):
        b = os.path.basename(f)
        m = CBG.search(b)
        if m:
            r = read(f)
            if r:
                r.update(method="CBG", cfg=f"γ={m.group(1)}", gamma=float(m.group(1)))
                rows.append(r)
            continue
        m = AD.search(b)
        if m:
            r = read(f)
            if r:
                C, rho, l0, lmax = m.groups()
                r.update(method="adual",
                         cfg=f"C=−log{CMAP.get(C, C)} ρ={rho} λ₀={l0} λmax={lmax}", gamma=None)
                rows.append(r)
    rows.sort(key=lambda x: x["viol"])

    floor_col = next((r["col"] for r in rows if r["method"] == "CBG" and r.get("gamma") == 0.0
                      and r["col"] is not None), 9.0)
    guard = floor_col + 3.0
    clean = [r for r in rows if r["col"] is not None and r["col"] <= guard]
    best_clean = clean[0] if clean else None

    os.makedirs(RES, exist_ok=True)
    L = ["# steps=1024 leaderboard (N=1000, toxic prefix, τ=0.5)\n",
         f"_Ranked by Viol@0.50. Fluency guard: collapse ≤ {guard:.0f}% (floor {floor_col:.0f}%+3). "
         "Low viol is often bought by degeneration — prefer the **fluency-clean best**._\n",
         "\n| Viol@0.50 | PPL | mean_tox | collapse% | distinct-1 | method | config |",
         "| --------: | --: | -------: | --------: | ---------: | :----- | :----- |"]
    for r in rows:
        flag = " ⚑" if (r["col"] is not None and r["col"] > guard) else ""
        c = f"{r['col']:.1f}%" if r["col"] is not None else "—"
        d = f"{r['d1']:.3f}" if r["d1"] is not None else "—"
        L.append(f"| {r['viol']:.1f}% | {r['ppl']:.1f} | {r['mt']:.3f} | {c}{flag} | {d} | {r['method']} | {r['cfg']} |")
    L.append("\n⚑ = collapse rate elevated (degeneration-driven low viol).\n")
    if rows:
        bv = rows[0]
        L.append(f"\n- **Lowest Viol (any):** {bv['viol']:.1f}% — {bv['method']} {bv['cfg']} "
                 f"(PPL {bv['ppl']:.1f}, collapse {bv['col']:.1f}%).")
    if best_clean:
        L.append(f"- **Fluency-clean best (collapse ≤ {guard:.0f}%):** {best_clean['viol']:.1f}% — "
                 f"{best_clean['method']} {best_clean['cfg']} (PPL {best_clean['ppl']:.1f}, "
                 f"collapse {best_clean['col']:.1f}%, distinct-1 {best_clean['d1']:.3f}).")
    L.append("\n> N=1000 ⇒ SE ≈ 1.5 pp on viol.\n")
    open(LB, "w").write("\n".join(L) + "\n")
    print(f"[summarize_1024] {len(rows)} runs -> {LB}")


if __name__ == "__main__":
    main()
