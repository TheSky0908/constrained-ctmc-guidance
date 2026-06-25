"""Plot λ and classifier-score trajectories for the INNER-LOOP J=4/n=4 runs
at steps=128 (ρ=0.2, λ₀=2, λ_max=5, time-dep clf, N=500), overlaying the two C:
  C=−log0.99 (0.01005, Viol@3.0=7.94%, Valid=75.6%)
  C=−log0.90 (0.10536, Viol@3.0=7.55%, Valid=76.8%)

Two-panel layout matching plot_lambda_traj_IL_J1n1.py:
  Left  : λ (median + 25–75% IQR band over the 500 trajectories)
  Right : mean log p(y|x) driving the dual update (median + 25–75% band, −C target line)
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = '/local/scratch/zhiheng/guidance'
RUNS = [
    dict(traj=f'{ROOT}/outputs/qm9/il_search/J4n4_C0p01005_rho0p2_l02_lmax5_steps128/samples.json.traj.json',
         label='C=−log0.99 (Viol 7.94%, Valid 75.6%)', color='tab:blue', cstr='−log0.99'),
    dict(traj=f'{ROOT}/outputs/qm9/il_search/J4n4_C0p10536_rho0p2_l02_lmax5_steps128/samples.json.traj.json',
         label='C=−log0.90 (Viol 7.55%, Valid 76.8%)', color='tab:red', cstr='−log0.90'),
]
OUT = f'{ROOT}/experiments/molecule_sa/figures/adual_IL_J4n4_timedep_traj_steps128.png'


def load(traj):
    d = json.load(open(traj))
    cfg = d['config']
    trajs = d['trajectories']
    K = len(trajs[0])
    B = len(trajs[0][0]['lambda'])
    N = len(trajs) * B
    lam = np.zeros((N, K))
    lp = np.zeros((N, K))
    for bi, batch in enumerate(trajs):
        for s, e in enumerate(batch):
            lam[bi * B:(bi + 1) * B, s] = e['lambda']
            lp[bi * B:(bi + 1) * B, s] = e['log_p_y']
    return cfg, lam, lp, K, N


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
lmax = None
K = None
for r in RUNS:
    cfg, lam, lp, K, N = load(r['traj'])
    lmax = cfg['gamma_lambda_max']
    steps = np.arange(K)
    c = r['color']
    # Left: λ
    ax1.plot(steps, np.median(lam, 0), color=c, lw=2, label=r['label'])
    ax1.fill_between(steps, np.percentile(lam, 25, 0), np.percentile(lam, 75, 0),
                     color=c, alpha=0.12)
    # Right: classifier score
    ax2.plot(steps, np.median(lp, 0), color=c, lw=2, label=f"median log p(y|x), {r['cstr']}")
    ax2.fill_between(steps, np.percentile(lp, 25, 0), np.percentile(lp, 75, 0),
                     color=c, alpha=0.12)
    ax2.axhline(-cfg['gamma_C'], color=c, ls='--', lw=1,
                label=f"−C = {r['cstr']} ≈ {-cfg['gamma_C']:.3f}")

ax1.axhline(lmax, color='gray', ls=':', lw=1, label=f'λ_max = {lmax:g}')
ax1.set_title('Inner-loop (Algm 2, J=4/n=4, time-dep clf): λ(t) @ 128 steps')
ax1.set_xlabel(f'Sampling step  (0 = start, all masked → {K - 1} = clean)')
ax1.set_ylabel('λ  (median, 25–75% band over 500 trajectories)')
ax1.set_ylim(0, lmax * 1.08)
ax1.legend()

ax2.set_title('Classifier score (drives λ update)')
ax2.set_xlabel('Sampling step')
ax2.set_ylabel('log p(y=1 | x_t)  (median, 25–75% band over 500)')
ax2.legend(fontsize=8)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches='tight')
print(f'wrote {OUT}  (K={K})')
