# Molecule SA-constrained generation (QM9)

Constrained generation of QM9 molecules under a **synthetic-accessibility (SA ≤ τ)** constraint,
reproducing and extending the CDD paper's molecular setting. This is the original domain in which
the adaptive-dual Lagrangian D-CBG method (the headline method of this repo) was developed.

## Layout
- `scripts/` — launchers
  - `train_sa_classifier.sh` — train the noisy DiT SA classifier (one per train-τ)
  - `sample_sa_dcbg.sh` — D-CBG, constant γ (baseline)
  - `sample_sa_dcbg_adaptive.sh` — **adaptive-dual** (the proposed method)
  - `sample_sa_dcbg_schedule.sh`, `run_gamma_schedules.sh` — static γ(t) schedules
  - `run_adaptive_dual*.sh` — adaptive-dual hyperparameter sweeps (v1–v6 iterations)
- `analysis/` — `plot_gamma_trajectories*.py`, `summarize_gamma_schedules.py`
- `figures/` — λ/γ trajectory plots
- `results/` — curated per-run summary CSVs (the reported numbers)
- `logs/` — archived run logs (git-ignored)
- `sa_dcbg_eval.md` — full evaluation writeup (D-CBG, static schedules, adaptive dual)
- `sampling_eval.md` — h-twist hard-constraint sampler evaluation
- `terminal.md` — copy-paste command log

## Driver & methods (at repo root)
- Eval driver: `sa_eval.py`
- Methods: D-CBG + adaptive-dual live in `diffusion.py` (`guidance=cbg`,
  `guidance.gamma_schedule=adaptive_dual`); hard-constraint h-twist samplers in
  `constrained_{ddpm,euler,fhs}.py`.

## Runtime outputs
Scripts write checkpoints / raw sample dumps under `outputs/qm9/...` and logs under `logs/`
at the repo root; both are git-ignored. The curated summaries in `results/` are the tracked
artifacts referenced by the writeups above.
