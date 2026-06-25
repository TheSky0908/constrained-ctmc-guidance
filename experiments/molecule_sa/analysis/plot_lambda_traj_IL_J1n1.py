"""Plot λ and classifier-score trajectories for the best INNER-LOOP run
(IL J=1/n=1, C=−log0.99, ρ=0.2, λ₀=2, λ_max=5, Viol@3.0=8.60%, steps=64, N=500).

Two-panel layout matching plot_gamma_trajectories_samplefirst_timedep.py:
  Left  : λ (median + 25–75% IQR band over the 500 trajectories)
  Right : mean log p(y|x) = classifier score driving the dual update
          (median + 25–75% band, with the −C target line)
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = '/local/scratch/zhiheng/guidance'
TRAJ = f'{ROOT}/outputs/qm9/run_innerloop_J1n1_timedep_n500_C0p01005/samples.json.traj.json'
OUT = f'{ROOT}/experiments/molecule_sa/figures/adual_IL_J1n1_timedep_traj_steps64_C0p01005.png'

d = json.load(open(TRAJ))
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
steps = np.arange(K)

color = 'tab:blue'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: λ
ax1.plot(steps, np.median(lam, 0), color=color, lw=2, label='median λ')
ax1.fill_between(steps, np.percentile(lam, 25, 0), np.percentile(lam, 75, 0),
                 color=color, alpha=0.15, label='25–75% band')
ax1.axhline(cfg['gamma_lambda_max'], color='gray', ls=':', lw=1,
            label=f"λ_max = {cfg['gamma_lambda_max']:g}")
ax1.set_title('Inner-loop (Algm 2, J=1/n=1, time-dep clf): λ(t) @ 64 steps  '
              '(C=−log0.99, Viol@3.0=8.60%)')
ax1.set_xlabel('Sampling step  (0 = start, all masked → 63 = clean)')
ax1.set_ylabel('λ  (median, 25–75% band over 500 trajectories)')
ax1.set_ylim(0, cfg['gamma_lambda_max'] * 1.08)
ax1.legend()

# Right: classifier score
ax2.plot(steps, np.median(lp, 0), color=color, lw=2, label='median log p(y|x)')
ax2.fill_between(steps, np.percentile(lp, 25, 0), np.percentile(lp, 75, 0),
                 color=color, alpha=0.15, label='25–75% band')
ax2.axhline(-cfg['gamma_C'], color='gray', ls='--',
            label=f"−C = −log0.99 ≈ {-cfg['gamma_C']:.3f}")
ax2.set_title('Classifier score (drives λ update)')
ax2.set_xlabel('Sampling step')
ax2.set_ylabel('log p(y=1 | x_t)  (median, 25–75% band over 500)')
ax2.legend()

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches='tight')
print(f'wrote {OUT}  (N={N}, K={K})')
