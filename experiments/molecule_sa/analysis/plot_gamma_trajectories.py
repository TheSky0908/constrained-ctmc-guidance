"""Plot γ(=λ) trajectories for adaptive_dual runs.

Parses log files left by sample_sa_dcbg_adaptive.sh, which print per-batch:
    step   0/127: mean_λ=X.XXX  max_λ=X.XXX  mean_log_p_y=±X.XXX
    step  32/127: ...
    step  64/127: ...
    step  96/127: ...
    step 127/127: ...

We average mean_λ at each step across all batches per run, then plot γ vs step
(step 0 = start of sampling at t=1, step 127 = end at t≈0).
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = '/local/scratch/zhiheng/guidance/logs'
OUT_DIR = '/local/scratch/zhiheng/guidance/figures'
os.makedirs(OUT_DIR, exist_ok=True)

RUNS = [
    ('C=−log 0.8, ρ=0.5', 'sa_dcbg_adual_n500_C0.2231_rho0.5_trainTau3.0.log',  'tab:blue'),
    ('C=−log 0.6, ρ=0.5', 'sa_dcbg_adual_n500_C0.5108_rho0.5_trainTau3.0.log',  'tab:orange'),
    ('C=−log 0.8, ρ=1.0', 'sa_dcbg_adual_n500_C0.2231_rho1.0_trainTau3.0.log',  'tab:green'),
]

LINE_RE = re.compile(
    r'step\s+(\d+)/(\d+):\s+mean_λ=([\-\d.]+)\s+max_λ=([\-\d.]+)\s+mean_log_p_y=([+\-\d.]+)'
)


def parse_log(path: str):
  """Return dict step -> list of (mean_λ, max_λ, mean_log_p)."""
  out = {}
  with open(path, 'r', errors='ignore') as f:
    for line in f:
      m = LINE_RE.search(line)
      if not m:
        continue
      step = int(m.group(1))
      mean_l = float(m.group(3))
      max_l = float(m.group(4))
      mean_lp = float(m.group(5))
      out.setdefault(step, []).append((mean_l, max_l, mean_lp))
  return out


def aggregate(parsed):
  steps = sorted(parsed.keys())
  mean_l = np.array([np.mean([x[0] for x in parsed[s]]) for s in steps])
  std_l = np.array([np.std([x[0] for x in parsed[s]]) for s in steps])
  max_l = np.array([np.mean([x[1] for x in parsed[s]]) for s in steps])
  mean_lp = np.array([np.mean([x[2] for x in parsed[s]]) for s in steps])
  return np.array(steps), mean_l, std_l, max_l, mean_lp


def main():
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

  # --- Plot 1: γ (mean λ) vs step, one line per config ---
  for label, fn, color in RUNS:
    path = os.path.join(LOG_DIR, fn)
    if not os.path.exists(path):
      print(f'Missing: {path}, skipping')
      continue
    parsed = parse_log(path)
    steps, mean_l, std_l, max_l, mean_lp = aggregate(parsed)
    n_batches = len(parsed[steps[0]])
    ax1.plot(steps, mean_l, '-o', color=color,
             label=f'{label}  (avg over {n_batches} batches)',
             linewidth=2, markersize=6)
    ax1.fill_between(steps, mean_l - std_l, mean_l + std_l,
                     color=color, alpha=0.15)
    print(f'{label}: steps={list(steps)} mean_λ={[f"{x:.3f}" for x in mean_l]}')

  ax1.set_xlabel('Sampling step  (0 = start, t≈1, all masked  →  127 = end, t≈0, clean)',
                 fontsize=10)
  ax1.set_ylabel('γ  =  mean λ  (averaged across batch + across all batches)', fontsize=10)
  ax1.set_title('Adaptive Dual Guidance — γ(t) trajectory', fontsize=12)
  ax1.legend(loc='upper right', fontsize=9)
  ax1.grid(True, alpha=0.3)
  ax1.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)

  # --- Plot 2: mean log p(y|x_t) vs step (mechanism check) ---
  for label, fn, color in RUNS:
    path = os.path.join(LOG_DIR, fn)
    if not os.path.exists(path):
      continue
    parsed = parse_log(path)
    steps, _, _, _, mean_lp = aggregate(parsed)
    ax2.plot(steps, mean_lp, '-o', color=color, label=label,
             linewidth=2, markersize=6)

  ax2.set_xlabel('Sampling step', fontsize=10)
  ax2.set_ylabel('mean log p(y=1 | x_t)', fontsize=10)
  ax2.set_title('Classifier score on partially-denoised x_t (drives γ update)',
                fontsize=12)
  # Threshold reference lines: −C means constraint satisfied with margin C
  for label, fn, _ in RUNS:
    c_match = re.search(r'C([\d.]+)', fn)
    if c_match:
      ax2.axhline(y=-float(c_match.group(1)), linestyle='--', alpha=0.3, color='gray')
  ax2.legend(loc='lower right', fontsize=9)
  ax2.grid(True, alpha=0.3)
  ax2.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)

  fig.tight_layout()
  out_path = os.path.join(OUT_DIR, 'adaptive_dual_gamma_trajectory.png')
  fig.savefig(out_path, dpi=130, bbox_inches='tight')
  print(f'wrote {out_path}')

  # Also save a single-panel γ plot for embedding in markdown
  fig2, ax = plt.subplots(figsize=(7, 4.5))
  for label, fn, color in RUNS:
    path = os.path.join(LOG_DIR, fn)
    if not os.path.exists(path):
      continue
    parsed = parse_log(path)
    steps, mean_l, std_l, _, _ = aggregate(parsed)
    ax.plot(steps, mean_l, '-o', color=color, label=label,
            linewidth=2, markersize=6)
    ax.fill_between(steps, mean_l - std_l, mean_l + std_l,
                    color=color, alpha=0.15)
  ax.set_xlabel('Sampling step  (start = 0, t=1  →  end = 127, t≈0)')
  ax.set_ylabel('γ = mean λ')
  ax.set_title('Adaptive Dual Guidance: γ(t) auto-adjusts per dual update')
  ax.legend()
  ax.grid(True, alpha=0.3)
  ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
  fig2.tight_layout()
  out2 = os.path.join(OUT_DIR, 'adaptive_dual_gamma_trajectory_single.png')
  fig2.savefig(out2, dpi=130, bbox_inches='tight')
  print(f'wrote {out2}')


if __name__ == '__main__':
  main()
