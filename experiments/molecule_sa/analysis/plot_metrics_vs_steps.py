"""Plot the remaining summary-table metrics vs sampling steps, from the top
table of sa_dcbg_adual_samplefirst_timedep.md (N=500, viol-best per method).

Companion to plot_viol_vs_steps.py: same methods/palette, but one panel per
metric (Valid / Unique / Novel / QED / Time), plus Viol@3.0 for reference.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS = [32, 64, 128, 256]
NAN = float("nan")

# metric -> {method: [val@32, @64, @128, @256]}
DATA = {
    "Valid (%) ↑": {
        "constant γ=0 (base)": [50.4, 59.6, 62.2, 64.2],
        "constant γ=1":        [60.8, 64.8, 70.8, 71.8],
        "constant γ=3":        [45.6, 65.0, 78.0, 80.6],
        "constant γ=5":        [ 2.6, 13.2, 31.0, 54.2],
        "adual sf+td (best)":  [64.0, 75.0, 82.0, 83.6],
        "adual IL+td (best)":  [65.6, 71.6, 78.4, 83.8],
    },
    "Unique (%) ↑": {
        "constant γ=0 (base)": [48.2, 58.8, 60.8, 61.0],
        "constant γ=1":        [46.0, 51.2, 57.6, 57.8],
        "constant γ=3":        [ 8.6, 25.8, 31.6, 36.4],
        "constant γ=5":        [ 0.6,  4.6,  8.6, 11.8],
        "adual sf+td (best)":  [35.0, 40.2, 46.6, 36.4],
        "adual IL+td (best)":  [34.8, 34.0, 36.4, 37.9],
    },
    "Novel & SA≤3.0 (count) ↑": {
        "constant γ=0 (base)": [12, 10,  8, 11],
        "constant γ=1":        [59, 46, 51, 55],
        "constant γ=3":        [ 7, 31, 43, 46],
        "constant γ=5":        [ 0,  5, 17, 25],
        "adual sf+td (best)":  [53, 42, 53, 56],
        "adual IL+td (best)":  [51, 31, 49, 64],
    },
    "QED ↑": {
        "constant γ=0 (base)": [0.419, 0.443, 0.468, 0.517],
        "constant γ=1":        [0.503, 0.493, 0.498, 0.510],
        "constant γ=3":        [0.409, 0.422, 0.444, 0.459],
        "constant γ=5":        [  NAN, 0.399, 0.417, 0.437],
        "adual sf+td (best)":  [0.461, 0.467, 0.480, 0.464],
        "adual IL+td (best)":  [0.469, 0.470, 0.447, 0.452],
    },
    "Sample time (s) ↓": {
        "constant γ=0 (base)": [337, 549, 794, 988],
        "constant γ=1":        [333, 543, 788, 977],
        "constant γ=3":        [325, 532, 767, 955],
        "constant γ=5":        [291, 468, 660, 826],
        "adual sf+td (best)":  [345, 574, 840, 1097],
        "adual IL+td (best)":  [403, 643, 1102, 1800],
    },
    "Viol @ 3.0 (%) ↓": {
        "constant γ=0 (base)": [83.33, 84.90, 86.50, 85.05],
        "constant γ=1":        [25.99, 25.62, 21.47, 20.89],
        "constant γ=3":        [43.86, 12.00,  8.21,  6.95],
        "constant γ=5":        [92.31, 34.85, 23.23,  9.96],
        "adual sf+td (best)":  [10.00,  9.33,  6.83,  3.35],
        "adual IL+td (best)":  [ 9.76,  7.26,  4.59,  2.86],
    },
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

fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.4))
fig.suptitle(
    "SA-constrained summary metrics vs sampling steps "
    "(N=500, viol-best per method; time-dependent classifier)",
    fontsize=13,
)

for ax, (metric, series) in zip(axes.flat, DATA.items()):
    for name, ys in series.items():
        c, m, ls, lw, z = STYLE[name]
        ax.plot(STEPS, ys, color=c, marker=m, linestyle=ls, linewidth=lw,
                markersize=7, label=name, zorder=z)
    ax.set_xscale("log", base=2)
    ax.set_xticks(STEPS)
    ax.set_xticklabels([str(s) for s in STEPS])
    ax.set_xlabel("sampling steps (log₂)")
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.grid(True, alpha=0.3)

# single shared legend
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.01))

fig.tight_layout(rect=[0, 0.04, 1, 0.95])
out = "experiments/molecule_sa/figures/metrics_vs_steps.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
