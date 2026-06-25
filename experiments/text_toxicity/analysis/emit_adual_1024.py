#!/usr/bin/env python3
"""Pick adaptive-dual configs for steps=1024 from the measured CBG steps=1024 curve.

Finds γ* = the CBG γ with lowest viol@0.5 at steps=1024 (the effective sweet spot,
which is high because of the dt dose-shift). Emits "C rho l0 lmax" lines for a
sweet-spot-start adual grid around γ*: warm-start λ₀ near γ*, cap λ_max just above,
vary ρ and C. Prints to stdout (one config per line). Falls back to a fixed high-γ
grid if no CBG runs are found.
"""
import csv, glob, os, re

OUT = "/local/scratch/zhiheng/guidance/outputs/owt/mdlm_owt"
CBG = re.compile(r"tox_cbg_gamma([\d.]+)_le0\.50_toxpfx_n1000_steps1024_results\.csv$")

best_g, best_v = None, 1e9
for f in glob.glob(os.path.join(OUT, "*_toxpfx_n1000_steps1024_results.csv")):
    m = CBG.search(os.path.basename(f))
    if not m:
        continue
    try:
        v = float(list(csv.DictReader(open(f)))[-1]["viol_tau_0.5"])
    except Exception:
        continue
    if v < best_v:
        best_v, best_g = v, float(m.group(1))

if best_g is None:
    best_g = 300.0  # fallback if CBG curve missing

g = int(round(best_g))
lo = max(1, int(round(g * 0.7)))
hi = int(round(g * 1.3))
hi2 = int(round(g * 1.6))
# C: 0.01005=−log0.99, 0.6931=−log0.5
cfgs = [
    f"0.6931 0.5 {g} {hi}",      # warm-start at γ*, cap above, reachable C, ρ=0.5
    f"0.01005 0.5 {g} {hi}",     # same, tight C
    f"0.6931 1.0 {g} {hi}",      # faster ramp
    f"0.6931 0.5 {lo} {hi}",     # start a bit lower
    f"0.6931 0.5 {g} {hi2}",     # higher cap (more headroom)
    f"0.01005 0.5 {hi} {hi2}",   # start high, cap higher (push effective γ up)
]
for c in cfgs:
    print(c)
