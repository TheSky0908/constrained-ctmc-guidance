"""Plot γ(=λ) and mean log p(y|x_t) trajectories for v6 adaptive_dual runs.

v6 = sampling.steps=2048, ρ=0.1, C ∈ {−log 0.99, 0}. Same plot style as v5.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = '/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance'
FIG_DIR = '/local/scratch/zhiheng/guidance/figures'
os.makedirs(FIG_DIR, exist_ok=True)

RUNS = [
    ('C=−log 0.99, ρ=0.1, steps=2048  ⭐ best Viol@3.0 (5.87%)',
     'mdlm_dcbg_sa_n500_C0.01005_rho0.1_l00.0_lmax50.0_trainTau3.0_steps2048_traj.json',
     'tab:purple'),
    ('C=0, ρ=0.1, steps=2048',
     'mdlm_dcbg_sa_n500_C0.0_rho0.1_l00.0_lmax50.0_trainTau3.0_steps2048_traj.json',
     'tab:blue'),
]


def load_full_traj(path: str):
  with open(path) as f:
    d = json.load(f)
  trajectories = d['trajectories']
  n_batches = len(trajectories)
  n_steps = len(trajectories[0])
  B = len(trajectories[0][0]['lambda'])
  n_total = n_batches * B
  lam = np.zeros((n_total, n_steps))
  lp = np.zeros((n_total, n_steps))
  for bi, batch in enumerate(trajectories):
    for s, entry in enumerate(batch):
      lam[bi * B:(bi + 1) * B, s] = np.array(entry['lambda'])
      lp[bi * B:(bi + 1) * B, s] = np.array(entry['log_p_y'])
  return np.arange(n_steps), lam, lp


def main():
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

  for label, fn, color in RUNS:
    path = os.path.join(OUT_DIR, fn)
    if not os.path.exists(path):
      print(f'Missing: {path}')
      continue
    steps, lam, lp = load_full_traj(path)
    n_total, n_steps = lam.shape
    mean_lam = lam.mean(axis=0)
    std_lam = lam.std(axis=0)
    mean_lp = lp.mean(axis=0)
    std_lp = lp.std(axis=0)

    print(f'\n{label}:  {n_total} trajectories × {n_steps} steps')
    for k in [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]:
      print(f'  step {k:5d}: γ = {mean_lam[k]:.3f} ± {std_lam[k]:.3f}  '
            f'log p = {mean_lp[k]:+.3f}')

    ax1.plot(steps, mean_lam, '-', color=color, linewidth=2, label=label)
    ax1.fill_between(steps, mean_lam - std_lam, mean_lam + std_lam,
                     color=color, alpha=0.15)

    ax2.plot(steps, mean_lp, '-', color=color, linewidth=2, label=label)
    ax2.fill_between(steps, mean_lp - std_lp, mean_lp + std_lp,
                     color=color, alpha=0.15)

  ax1.set_xlabel(
      'Sampling step  (0 = start, t=1, all masked  →  2047 = end, t≈0, clean)',
      fontsize=10)
  ax1.set_ylabel('γ = λ  (mean ± std over 500 trajectories)', fontsize=10)
  ax1.set_title(
      'Adaptive Dual γ(t) at sampling.steps=2048, ρ=0.1',
      fontsize=11)
  ax1.legend(loc='lower right', fontsize=9)
  ax1.grid(True, alpha=0.3)
  ax1.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)

  ax2.set_xlabel('Sampling step', fontsize=10)
  ax2.set_ylabel('mean log p(y=1 | x_t)  (mean ± std over 500)', fontsize=10)
  ax2.set_title('Classifier score (drives γ update)', fontsize=11)
  ax2.axhline(y=-0.01005, linestyle='--', alpha=0.5, color='tab:purple',
              label='−C = −log 0.99 ≈ −0.010')
  ax2.axhline(y=0, linestyle='--', alpha=0.5, color='tab:blue',
              label='−C = 0  (C=0)')
  ax2.legend(loc='lower right', fontsize=9)
  ax2.grid(True, alpha=0.3)

  fig.tight_layout()
  out_path = os.path.join(FIG_DIR, 'adaptive_dual_gamma_trajectory_v6.png')
  fig.savefig(out_path, dpi=130, bbox_inches='tight')
  print(f'\nwrote {out_path}')


if __name__ == '__main__':
  main()
