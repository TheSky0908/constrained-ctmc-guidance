#!/usr/bin/env python3
"""Summarize an adaptive-dual tuning sweep on the matched diagonal (train τ = eval τ).

τ is chosen by the TUNE_TAU env var ("0.25" | "0.50" | "0.75", default "0.50").
Reads every matching result CSV in outputs/owt/mdlm_owt, parses the config from
the filename, and ranks configs against the CBG gamma=3 baseline. Goal: lowest
Viol@τ; tie-break / secondary objective = lowest PPL among configs that beat the
CBG gamma=3 viol. Writes results/tuning_leaderboard[_tau<τ>].md. Idempotent.
"""
import csv, glob, os, re, datetime

ROOT = "/local/scratch/zhiheng/guidance"
OUT_DIR = os.path.join(ROOT, "outputs/owt/mdlm_owt")
RES_DIR = os.path.join(ROOT, "experiments/text_toxicity/results")

FILE_TAU = os.environ.get("TUNE_TAU", "0.50")     # "0.25" | "0.50" | "0.75"
COL = "viol_tau_" + str(float(FILE_TAU))           # 0.50 -> viol_tau_0.5
LB = os.path.join(RES_DIR, "tuning_leaderboard.md" if FILE_TAU == "0.50"
                  else f"tuning_leaderboard_tau{FILE_TAU}.md")

ADUAL_RE = re.compile(
    rf"tox_adual_C([-\d.]+)_rho([\d.]+)_l0([\d.]+)_lmax([\d.]+)_le{FILE_TAU}_n500_steps128_results\.csv$")
CBG_RE = re.compile(rf"tox_cbg_gamma([\d.]+)_le{FILE_TAU}_n500_steps128_results\.csv$")


def read_row(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    r = rows[-1]
    try:
        viol = float(r[COL]); ppl = float(r["ppl_gpt2xl"])
        mtox = float(r["mean_toxicity"]); n = int(float(r["n"]))
    except (KeyError, ValueError, TypeError):
        return None
    return dict(viol=viol, ppl=ppl, mtox=mtox, n=n)


def collect():
    cbg, adual = [], []
    for path in glob.glob(os.path.join(OUT_DIR, "*_results.csv")):
        base = os.path.basename(path)
        m = CBG_RE.search(base)
        if m:
            row = read_row(path)
            if row:
                row.update(kind="cbg", gamma=float(m.group(1)), label=f"CBG γ={m.group(1)}")
                cbg.append(row)
            continue
        m = ADUAL_RE.search(base)
        if m:
            row = read_row(path)
            if row:
                C, rho, l0, lmax = m.groups()
                row.update(kind="adual", C=float(C), rho=float(rho), l0=float(l0), lmax=float(lmax),
                           label=f"adual C={C} ρ={rho} λ₀={l0} λmax={lmax}")
                adual.append(row)
    return cbg, adual


def main():
    cbg, adual = collect()
    target = next((r["viol"] for r in cbg if abs(r["gamma"] - 3.0) < 1e-9), None)
    os.makedirs(RES_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = []
    L.append(f"# Adaptive-dual tuning leaderboard (τ={FILE_TAU}, steps=128, N=500)\n")
    L.append(f"_Auto-generated {ts}. Goal: beat CBG γ=3 on Viol@{FILE_TAU}, then minimize PPL._\n")
    L.append(f"**Target = CBG γ=3 Viol@{FILE_TAU} = {100*target:.2f}%**\n" if target is not None
             else "**Target = CBG γ=3 — not run yet.**\n")
    L.append(f"\nRuns finished: {len(cbg)} CBG + {len(adual)} adaptive-dual = {len(cbg)+len(adual)}.\n")

    L.append("\n## CBG baselines (constant γ)\n")
    L.append(f"| γ | N | Viol@{FILE_TAU} | PPL | mean_tox |")
    L.append("| --: | --: | --: | --: | --: |")
    for r in sorted(cbg, key=lambda x: x["gamma"]):
        L.append(f"| {r['gamma']:g} | {r['n']} | {100*r['viol']:.2f}% | {r['ppl']:.2f} | {r['mtox']:.3f} |")

    L.append("\n## Adaptive-dual configs (ranked: beats-target first by PPL, then by Viol)\n")
    def sort_key(r):
        beats = (target is not None and r["viol"] < target - 1e-12)
        return (0 if beats else 1, r["ppl"] if beats else r["viol"], r["viol"], r["ppl"])
    L.append(f"| ✓ | C | ρ | λ₀ | λmax | N | Viol@{FILE_TAU} | PPL | mean_tox |")
    L.append("| :-: | --: | --: | --: | --: | --: | --: | --: | --: |")
    winners = []
    for r in sorted(adual, key=sort_key):
        beats = (target is not None and r["viol"] < target - 1e-12)
        if beats: winners.append(r)
        L.append(f"| {'✅' if beats else ''} | {r['C']:g} | {r['rho']:g} | {r['l0']:g} | {r['lmax']:g} | {r['n']} | "
                 f"{100*r['viol']:.2f}% | {r['ppl']:.2f} | {r['mtox']:.3f} |")

    L.append("\n## Verdict\n")
    if target is None:
        L.append("- CBG γ=3 baseline not finished yet — no target to compare against.\n")
    elif not winners:
        best = min(adual, key=lambda x: x["viol"]) if adual else None
        L.append(f"- **No adaptive-dual config beats CBG γ=3 ({100*target:.2f}%) yet.**")
        if best:
            L.append(f"- Closest: {best['label']} → Viol@{FILE_TAU} {100*best['viol']:.2f}%, PPL {best['ppl']:.2f}.")
    else:
        best = min(winners, key=lambda x: x["ppl"])
        L.append(f"- **{len(winners)} config(s) beat CBG γ=3 on Viol@{FILE_TAU}.**")
        L.append(f"- **Best (lowest PPL among winners):** {best['label']}")
        L.append(f"  → Viol@{FILE_TAU} **{100*best['viol']:.2f}%** (vs {100*target:.2f}%), PPL **{best['ppl']:.2f}**, mean_tox {best['mtox']:.3f}.")
        bv = min(winners, key=lambda x: x["viol"])
        L.append(f"- Lowest-viol winner: {bv['label']} → Viol@{FILE_TAU} {100*bv['viol']:.2f}%, PPL {bv['ppl']:.2f}.")
    L.append("\n> N=500 ⇒ SE on a ~25% rate ≈ 1.9 pp; treat sub-2pp gaps as ties.\n")

    with open(LB, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[summarize τ={FILE_TAU}] wrote {LB}  ({len(cbg)} cbg + {len(adual)} adual)")


if __name__ == "__main__":
    main()
