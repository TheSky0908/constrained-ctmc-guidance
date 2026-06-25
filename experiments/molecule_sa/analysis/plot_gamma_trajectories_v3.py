"""Plot γ(=λ) and mean log p(y|x_t) trajectories for v3 adaptive_dual runs.

These v3 runs (C=−log 0.99, ρ=0.2 / ρ=0.5) were sampled with the new code that
dumps the full 128-step, per-sample trajectory to <run>_traj.json. So we can
plot at full resolution (128 points) instead of the 5-point log resolution used
for v1.

Aggregation: at each step k, we have n_batches × B per-sample (λ, log_p_y)
values. We report mean ± std across all (n_batches × B) samples = 125 × 4 = 500
trajectories.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = '/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance'
FIG_DIR = '/local/scratch/zhiheng/guidance/figures'
os.makedirs(FIG_DIR, exist_ok=True)

RUNS = [
    ('C=−log 0.99, ρ=0.2  ⭐ best Viol@3.0',
     'mdlm_dcbg_sa_n500_C0.01005_rho0.2_l00.0_lmax50.0_trainTau3.0_traj.json',
     'tab:purple'),
    ('C=−log 0.99, ρ=0.5',
     'mdlm_dcbg_sa_n500_C0.01005_rho0.5_l00.0_lmax50.0_trainTau3.0_traj.json',
     'tab:red'),
]


def load_full_traj(path: str):
  """Return (steps, t_norms, lambda_arr (n_total, n_steps),
  log_p_arr (n_total, n_steps))."""
  with open(path) as f:
    d = json.load(f)
  trajectories = d['trajectories']
  n_batches = len(trajectories)
  n_steps = len(trajectories[0])
  B = len(trajectories[0][0]['lambda'])  # batch size
  n_total = n_batches * B
  lam = np.zeros((n_total, n_steps))
  lp = np.zeros((n_total, n_steps))
  t_norms = np.array([trajectories[0][s]['t_norm'] for s in range(n_steps)])
  for bi, batch in enumerate(trajectories):
    for s, entry in enumerate(batch):
      lam[bi * B:(bi + 1) * B, s] = np.array(entry['lambda'])
      lp[bi * B:(bi + 1) * B, s] = np.array(entry['log_p_y'])
  return np.arange(n_steps), t_norms, lam, lp


def main():
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

  for label, fn, color in RUNS:
    path = os.path.join(OUT_DIR, fn)
    if not os.path.exists(path):
      print(f'Missing: {path}, skipping')
      continue

    steps, t_norms, lam, lp = load_full_traj(path)
    n_total = lam.shape[0]

    # Aggregate per-step
    mean_lam = lam.mean(axis=0)
    std_lam = lam.std(axis=0)
    mean_lp = lp.mean(axis=0)
    std_lp = lp.std(axis=0)

    print(f'\n{label}:  {n_total} trajectories × {len(steps)} steps')
    print(f'  step 0  : γ = {mean_lam[0]:.3f} ± {std_lam[0]:.3f}')
    print(f'  step 32 : γ = {mean_lam[32]:.3f} ± {std_lam[32]:.3f}')
    print(f'  step 64 : γ = {mean_lam[64]:.3f} ± {std_lam[64]:.3f}')
    print(f'  step 96 : γ = {mean_lam[96]:.3f} ± {std_lam[96]:.3f}')
    print(f'  step 127: γ = {mean_lam[127]:.3f} ± {std_lam[127]:.3f}')

    # Panel 1: γ vs step
    ax1.plot(steps, mean_lam, '-', color=color, linewidth=2, label=label)
    ax1.fill_between(steps, mean_lam - std_lam, mean_lam + std_lam,
                     color=color, alpha=0.15)

    # Panel 2: mean log p(y|x_t) vs step
    ax2.plot(steps, mean_lp, '-', color=color, linewidth=2, label=label)
    ax2.fill_between(steps, mean_lp - std_lp, mean_lp + std_lp,
                     color=color, alpha=0.15)

  ax1.set_xlabel(
      'Sampling step  (0 = start, t=1, all masked  →  127 = end, t≈0, clean)',
      fontsize=10)
  ax1.set_ylabel('γ = λ  (mean ± std over 500 trajectories)', fontsize=10)
  ax1.set_title(
      'Adaptive Dual: γ(t) at full 128-step resolution  (C = −log 0.99)',
      fontsize=11)
  ax1.legend(loc='upper right', fontsize=9)
  ax1.grid(True, alpha=0.3)
  ax1.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)

  ax2.set_xlabel('Sampling step', fontsize=10)
  ax2.set_ylabel('mean log p(y=1 | x_t)  (mean ± std over 500)', fontsize=10)
  ax2.set_title('Classifier score (drives γ update)', fontsize=11)
  # Threshold reference: −C = −0.01005 (very close to 0; constraint is "p ≥ 0.99")
  ax2.axhline(y=-0.01005, linestyle='--', alpha=0.5, color='gray',
              label='−C = −log 0.99 ≈ −0.010')
  ax2.legend(loc='lower right', fontsize=9)
  ax2.grid(True, alpha=0.3)
  ax2.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)

  fig.tight_layout()
  out_path = os.path.join(FIG_DIR, 'adaptive_dual_gamma_trajectory_v3.png')
  fig.savefig(out_path, dpi=130, bbox_inches='tight')
  print(f'\nwrote {out_path}')


if __name__ == '__main__':
  main()
