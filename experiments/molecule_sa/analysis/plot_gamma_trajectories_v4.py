"""Plot γ(=λ) and mean log p(y|x_t) trajectories for v4 adaptive_dual runs.

v4 = extreme C: C=0 (require p=1, unreachable) and C=-log 1.01 (require p>1,
strictly impossible). 4 configs total (2 C × 2 ρ). All have full per-step
trajectories in <run>_traj.json (125 batches × 128 steps × 4 samples).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = '/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance'
FIG_DIR = '/local/scratch/zhiheng/guidance/figures'
os.makedirs(FIG_DIR, exist_ok=True)

RUNS = [
    ('C=0, ρ=0.2',
     'mdlm_dcbg_sa_n500_C0.0_rho0.2_l00.0_lmax50.0_trainTau3.0_traj.json',
     'tab:blue', '-'),
    ('C=0, ρ=0.5',
     'mdlm_dcbg_sa_n500_C0.0_rho0.5_l00.0_lmax50.0_trainTau3.0_traj.json',
     'tab:blue', '--'),
    ('C=−log 1.01, ρ=0.2',
     'mdlm_dcbg_sa_n500_C-0.00995_rho0.2_l00.0_lmax50.0_trainTau3.0_traj.json',
     'tab:red', '-'),
    ('C=−log 1.01, ρ=0.5  ⭐ best Viol@3.0',
     'mdlm_dcbg_sa_n500_C-0.00995_rho0.5_l00.0_lmax50.0_trainTau3.0_traj.json',
     'tab:red', '--'),
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
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

  for label, fn, color, ls in RUNS:
    path = os.path.join(OUT_DIR, fn)
    if not os.path.exists(path):
      print(f'Missing: {path}, skipping')
      continue
    steps, lam, lp = load_full_traj(path)
    n_total = lam.shape[0]
    mean_lam = lam.mean(axis=0)
    std_lam = lam.std(axis=0)
    mean_lp = lp.mean(axis=0)
    std_lp = lp.std(axis=0)

    print(f'\n{label}:  {n_total} trajectories × {len(steps)} steps')
    for k in [0, 32, 64, 96, 127]:
      print(f'  step {k:3d}: γ = {mean_lam[k]:.3f} ± {std_lam[k]:.3f}'
            f'    log p(y|x_t) = {mean_lp[k]:+.3f}')

    ax1.plot(steps, mean_lam, ls, color=color, linewidth=2, label=label)
    ax1.fill_between(steps, mean_lam - std_lam, mean_lam + std_lam,
                     color=color, alpha=0.1)

    ax2.plot(steps, mean_lp, ls, color=color, linewidth=2, label=label)
    ax2.fill_between(steps, mean_lp - std_lp, mean_lp + std_lp,
                     color=color, alpha=0.1)

  ax1.set_xlabel('Sampling step  (0 = start, t=1, all masked  →  127 = end, t≈0, clean)',
                 fontsize=10)
  ax1.set_ylabel('γ = λ  (mean ± std over 500 trajectories)', fontsize=10)
  ax1.set_title(
      'Adaptive Dual: γ(t) at full 128-step resolution  (C ∈ {0, −log 1.01}, extreme)',
      fontsize=11)
  ax1.legend(loc='lower right', fontsize=9)
  ax1.grid(True, alpha=0.3)
  ax1.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)

  ax2.set_xlabel('Sampling step', fontsize=10)
  ax2.set_ylabel('mean log p(y=1 | x_t)  (mean ± std over 500)', fontsize=10)
  ax2.set_title('Classifier score (drives γ update)', fontsize=11)
  ax2.axhline(y=0, linestyle='--', alpha=0.5, color='tab:blue',
              label='−C = 0  (C=0)')
  ax2.axhline(y=0.00995, linestyle='--', alpha=0.5, color='tab:red',
              label='−C = +0.0100  (C=−log 1.01)')
  ax2.legend(loc='lower right', fontsize=9)
  ax2.grid(True, alpha=0.3)

  fig.tight_layout()
  out_path = os.path.join(FIG_DIR, 'adaptive_dual_gamma_trajectory_v4.png')
  fig.savefig(out_path, dpi=130, bbox_inches='tight')
  print(f'\nwrote {out_path}')


if __name__ == '__main__':
  main()
