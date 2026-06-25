"""Plot Viol@3.0 vs sampling steps, from the summary table at the top of
sa_dcbg_adual_samplefirst_timedep.md (N=500, viol-best per method).

Style mirrors the constant-γ frontier figure: two panels, red/blue palette,
labelled markers. Panel A = all methods (full Viol range); Panel B = zoom on
the low-violation regime (the methods that actually reach the <12% target).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS = [32, 64, 128, 256]

# Viol@3.0 (%), viol-best row per method, from the K=32/64/128/256 summary blocks.
VIOL = {
    "constant γ=0 (base)": [83.33, 84.90, 86.50, 85.05],
    "constant γ=1":        [25.99, 25.62, 21.47, 20.89],
    "constant γ=3":        [43.86, 12.00,  8.21,  6.95],
    "constant γ=5":        [92.31, 34.85, 23.23,  9.96],
    "adual sf+td (best)":  [10.00,  9.33,  6.83,  3.35],
    "adual IL+td (best)":  [ 9.76,  7.26,  4.59,  2.86],
}

# (color, marker, linestyle, linewidth, zorder)
STYLE = {
    "constant γ=0 (base)": ("#9aa0a6", "o", "--", 1.6, 1),
    "constant γ=1":        ("#5b8ff9", "s", "-",  1.8, 2),
    "constant γ=3":        ("#5ad8a6", "^", "-",  1.8, 2),
    "constant γ=5":        ("#f6bd16", "D", "--", 1.6, 1),
    "adual sf+td (best)":  ("#e8684a", "o", "-",  2.4, 4),
    "adual IL+td (best)":  ("#c0392b", "P", "-",  2.6, 5),
}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 5.0))
fig.suptitle(
    "SA-constrained Viol@3.0 vs sampling steps "
    "(N=500, viol-best per method; time-dependent classifier)",
    fontsize=12,
)

# ---- Panel A: all methods, full Viol range ----
for name, ys in VIOL.items():
    c, m, ls, lw, z = STYLE[name]
    axA.plot(STEPS, ys, color=c, marker=m, linestyle=ls, linewidth=lw,
             markersize=7, label=name, zorder=z)
axA.set_xscale("log", base=2)
axA.set_xticks(STEPS)
axA.set_xticklabels([str(s) for s in STEPS])
axA.set_xlabel("sampling steps (log₂)")
axA.set_ylabel("Viol @ 3.0  (%, lower better)")
axA.set_title("(A) all methods — full range")
axA.grid(True, alpha=0.3)
axA.legend(fontsize=8.5, loc="upper right")

# ---- Panel B: zoom on the low-violation regime ----
ZOOM = ["constant γ=3", "constant γ=5", "adual sf+td (best)", "adual IL+td (best)"]
for name in ZOOM:
    ys = VIOL[name]
    c, m, ls, lw, z = STYLE[name]
    axB.plot(STEPS, ys, color=c, marker=m, linestyle=ls, linewidth=lw,
             markersize=7, label=name, zorder=z)
    for x, y in zip(STEPS, ys):
        axB.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=7.5, color=c)
axB.set_xscale("log", base=2)
axB.set_xticks(STEPS)
axB.set_xticklabels([str(s) for s in STEPS])
axB.set_xlabel("sampling steps (log₂)")
axB.set_ylabel("Viol @ 3.0  (%, lower better)")
axB.set_title("(B) low-violation regime — zoom")
axB.set_ylim(0, 48)
axB.axhline(12.0, color="gray", linestyle=":", linewidth=1.0)
axB.text(STEPS[0], 12.6, "12% target", fontsize=7.5, color="gray")
axB.grid(True, alpha=0.3)
axB.legend(fontsize=8.5, loc="upper right")

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "experiments/molecule_sa/figures/viol_vs_steps.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
