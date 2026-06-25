#!/usr/bin/env python3
"""Plot the adaptive-dual λ(=γ) trajectory over sampling steps from a run's
*_traj.json. λ on the left axis (mean ± std band across all samples + %-at-cap),
mean log p(y|x_t) on the right axis.

Usage: python plot_lambda_trajectory.py <traj.json> <out.png> [lambda_max]
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

traj_path = sys.argv[1]
out_path = sys.argv[2]
lmax = float(sys.argv[3]) if len(sys.argv) > 3 else None

batches = json.load(open(traj_path))["trajectories"]   # [batch][step]{lambda,log_p_y,t_norm}
n_steps = len(batches[0])

lam_mean, lam_std, lp_mean, atcap, tnorm = [], [], [], [], []
for s in range(n_steps):
    lam, lp = [], []
    for b in batches:
        lam.extend(b[s]["lambda"]); lp.extend(b[s]["log_p_y"])
    lam = np.array(lam, float); lp = np.array(lp, float)
    lam_mean.append(lam.mean()); lam_std.append(lam.std())
    lp_mean.append(lp.mean()); tnorm.append(batches[0][s]["t_norm"])
    if lmax: atcap.append(100 * np.mean(lam >= lmax - 1e-3))
x = np.arange(n_steps)
lam_mean = np.array(lam_mean); lam_std = np.array(lam_std)

fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.plot(x, lam_mean, color="C0", lw=2, label="λ (mean)")
ax1.fill_between(x, lam_mean - lam_std, lam_mean + lam_std, color="C0", alpha=0.2,
                 label="±1 std")
if lmax:
    ax1.axhline(lmax, color="C3", ls="--", lw=1, label=f"λ_max={lmax:g}")
ax1.set_xlabel("sampling step (0 = fully masked, t→0 clean)")
ax1.set_ylabel("λ (= γ)", color="C0")
ax1.tick_params(axis="y", labelcolor="C0")

ax2 = ax1.twinx()
ax2.plot(x, lp_mean, color="C2", lw=1.5, ls="-.", label="mean log p(y|xₜ)")
ax2.set_ylabel("mean log p(y|xₜ)", color="C2")
ax2.tick_params(axis="y", labelcolor="C2")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)
ax1.set_title(out_path.split("/")[-1].replace(".png", ""))
plt.tight_layout()
plt.savefig(out_path, dpi=130)

# brief textual summary for the caller
pts = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
print("step | t_norm | λ mean±std | log p" + (" | %@cap" if lmax else ""))
for s in pts:
    line = f"{s:4d} | {tnorm[s]:.3f} | {lam_mean[s]:.2f}±{lam_std[s]:.2f} | {lp_mean[s]:.2f}"
    if lmax:
        line += f" | {atcap[s]:.0f}%"
    print(line)
print("saved ->", out_path)
