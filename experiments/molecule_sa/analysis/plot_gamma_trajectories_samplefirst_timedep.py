"""Plot γ(=λ) and mean log p(y|x_t) trajectories for the adaptive-dual
SAMPLE_FIRST + TIME-DEPENDENT-classifier sweep (steps=128, N=500, λ₀=2, λ_max=5).

Mirrors plot_gamma_trajectories_v3.py but points at the new run filenames
  mdlm_dcbg_sa_adual_sf_td_C<C>_rho<RHO>_l02.0_lmax5.0_trainTau3.0_traj.json
and produces ONE 2-panel figure per C value, overlaying ρ ∈ {0.1, 0.2, 0.5}.

Left panel : γ=λ (mean ± std over 500 trajectories) vs sampling step.
Right panel: mean log p(y=1|x_t) (the score driving the dual update).

Note: in sample_first mode the logged log_p_y is evaluated at the NEWLY drawn
state x_{t+1} (at its own noise level σ_s), and λ is the value used to DRAW that
step (pre-update) — see diffusion.Diffusion._diffusion_sample (sample_first).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = '/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance'
FIG_DIR = '/local/scratch/zhiheng/guidance/experiments/molecule_sa/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# (C string in filename, pretty label for C)
C_VALUES = [
    ('0.2231', '−log 0.8'),
    ('0.01005', '−log 0.99'),
    ('-0.00995', '−log 1.01'),
]
RHOS = [('0.1', 'tab:blue'), ('0.2', 'tab:purple'), ('0.5', 'tab:red')]
L0, LMAX, TAU = '2.0', '5.0', '3.0'


def fn_for(c, rho):
  return (f'mdlm_dcbg_sa_adual_sf_td_C{c}_rho{rho}'
          f'_l0{L0}_lmax{LMAX}_trainTau{TAU}_traj.json')


def load_full_traj(path):
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
  for c, c_label in C_VALUES:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    any_plotted = False
    for rho, color in RHOS:
      path = os.path.join(OUT_DIR, fn_for(c, rho))
      if not os.path.exists(path):
        print(f'Missing: {path}, skipping')
        continue
      any_plotted = True
      steps, lam, lp = load_full_traj(path)
      # Median line + 25–75% (IQR) band over the 500 trajectories. The IQR band
      # respects the one-sided cap of log p ≤ 0 (no spurious >0 region that a
      # symmetric mean±std band produces near saturation).
      lam_med = np.median(lam, 0)
      lam_q1, lam_q3 = np.percentile(lam, 25, 0), np.percentile(lam, 75, 0)
      lp_med = np.median(lp, 0)
      lp_q1, lp_q3 = np.percentile(lp, 25, 0), np.percentile(lp, 75, 0)
      lbl = f'C={c_label}, ρ={rho}'
      ax1.plot(steps, lam_med, color=color, label=lbl, lw=2)
      ax1.fill_between(steps, lam_q1, lam_q3, color=color, alpha=0.15)
      ax2.plot(steps, lp_med, color=color, label=lbl, lw=2)
      ax2.fill_between(steps, lp_q1, lp_q3, color=color, alpha=0.15)
    if not any_plotted:
      plt.close(fig)
      continue
    # -C reference line on the right panel.
    try:
      neg_C = -float(c)
      ax2.axhline(neg_C, color='gray', ls='--',
                  label=f'−C = {c_label} ≈ {neg_C:.3f}')
    except ValueError:
      pass
    ax1.set_title(f'Adaptive Dual (sample_first, time-dep clf): '
                  f'γ(t) @ 128 steps  (C = {c_label})')
    ax1.set_xlabel('Sampling step  (0 = start, all masked → 127 = clean)')
    ax1.set_ylabel('γ = λ  (median, 25–75% band over 500 trajectories)')
    ax1.legend()
    ax2.set_title('Classifier score (drives γ update)')
    ax2.set_xlabel('Sampling step')
    ax2.set_ylabel('log p(y=1 | x_t)  (median, 25–75% band over 500)')
    ax2.legend()
    fig.tight_layout()
    safe_c = c.replace('-', 'm').replace('.', 'p')
    out = os.path.join(FIG_DIR, f'adual_samplefirst_timedep_traj_C{safe_c}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')
    plt.close(fig)


if __name__ == '__main__':
  main()
