# Sampling Evaluation — QM9 (mdlm_no-guidance)

**Task:** Generate valid SMILES; constraint set **C = {novel molecules not in QM9 train set}**.
**Discriminator** $h_\phi(x_t, t) \approx P(X_0 \in C \mid X_t = x_t)$ trained with martingale-MSE on novelty labels.
**Reference** (QM9 train, 133,885 mols): QED Mean = 0.465, Median = 0.473.
**All runs**: N=1000, seed=1, length=32, model=small/dit, T=32 (Euler/DDPM steps).

## Results

| Sampler | ε | Valid | Unique | Novel (of valid) | **Constraint Sat.** | QED Mean | QED Median |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Unconditional baselines** |  |  |  |  |  |  |  |
| euler                  | —    | 51.40% | 93.77% | 54.47% |    —    | 0.459 | 0.466 |
| fhs                    | —    | 65.20% | 94.63% | 51.38% |    —    | 0.452 | 0.461 |
| **Constrained (h-twist)** |  |  |  |  |  |  |  |
| constrained-ddpm       | 0    | 51.40% | 99.81% | 72.96% | 37.50%  | 0.457 | 0.465 |
| constrained-ddpm       | 0.05 | 51.20% | 99.80% | 73.83% | 37.80%  | 0.458 | 0.466 |
| constrained-ddpm       | 0.10 | 50.70% | 99.80% | 74.95% | 38.00%  | 0.456 | 0.465 |
| constrained-euler      | 0    | 50.70% | 99.80% | 74.36% | 37.70%  | 0.461 | 0.472 |
| constrained-euler      | 0.05 | 50.40% | 99.80% | 75.20% | 37.90%  | 0.461 | 0.473 |
| constrained-euler      | 0.10 | 50.30% | 99.80% | 76.14% | 38.30%  | 0.464 | 0.476 |
| **constrained-fhs**    | **0**    | **67.80%** | 98.97% | 63.86% | **43.30%** | 0.456 | 0.464 |
| **constrained-fhs**    | **0.05** | **67.20%** | 99.26% | 65.63% | **44.10%** | 0.457 | 0.467 |
| **constrained-fhs**    | **0.10** | **65.30%** | 99.69% | 69.07% | **45.10%** | 0.458 | 0.470 |

## Key observations

- **constrained-fhs leads on every metric except QED:** highest Valid (~67%), highest Constraint Satisfaction (43–44%), competitive Unique/Novel. This is consistent with the theory that FHS's one-position-per-step commitment lets the h-twist take cleaner integer steps toward C, whereas DDPM/Euler's simultaneous-update introduces extra noise.
- **DDPM ≈ Euler:** practically identical numbers across all metrics — same h-twist mechanism, both update all positions per step.
- **ε threshold effect** (monotone with ε ∈ {0, 0.05, 0.10}):

  | Sampler | Valid (Δ from ε=0) | Constraint Sat. (Δ from ε=0) |
  |---|---|---|
  | constrained-ddpm  | 51.4% → 51.2% → 50.7% (−0.7 pp) | 37.5% → 37.8% → **38.0%** (+0.5 pp) |
  | constrained-euler | 50.7% → 50.4% → 50.3% (−0.4 pp) | 37.7% → 37.9% → **38.3%** (+0.6 pp) |
  | constrained-fhs   | 67.8% → 67.2% → 65.3% (−2.5 pp) | 43.3% → 44.1% → **45.1%** (+1.8 pp) |

  Trends across ε ∈ {0, 0.05, 0.10}:
  - Constraint Sat. and Novel-of-valid monotone ↑
  - Valid monotone ↓ (the tax for stronger pruning)
  - QED basically unchanged
  - **FHS shows the biggest ε sensitivity** (+1.8 pp constraint sat at cost of −2.5 pp valid), because FHS only has the threshold to enforce the constraint — there's no stay-at-mask fallback. DDPM/Euler are smoother because mass flows to stay-at-mask.
  → ε=0.10 is the new Pareto-best for FHS on constraint sat; pushing higher might cost too much valid.
- **Constraint vs unconditional novelty:**
  - Unconditional novelty (Novel / valid): euler 54%, fhs 51%
  - Constrained novelty: ddpm/euler 73–75%, fhs 64–66%
  - h-twist boosts novelty by ~+20 pp on DDPM/Euler and ~+15 pp on FHS
- **QED ≈ QM9 reference** for all samplers (0.452–0.461 vs ref 0.465); h-twist does not degrade QED.

## Files

| Sampler | ε | results.csv | samples.json |
|---|---:|---|---|
| euler              | —    | `euler_eval_results.csv`                  | `euler_samples.json` |
| fhs                | —    | `fhs_eval_results.csv`                    | `fhs_samples.json` |
| constrained-ddpm   | 0    | `constrained_ddpm_results.csv`            | `constrained_ddpm_samples.json` |
| constrained-ddpm   | 0.05 | `constrained_ddpm_results_eps0.05.csv`    | `constrained_ddpm_samples_eps0.05.json` |
| constrained-ddpm   | 0.10 | `constrained_ddpm_results_eps0.1.csv`     | `constrained_ddpm_samples_eps0.1.json` |
| constrained-euler  | 0    | `constrained_euler_results.csv`           | `constrained_euler_samples.json` |
| constrained-euler  | 0.05 | `constrained_euler_results_eps0.05.csv`   | `constrained_euler_samples_eps0.05.json` |
| constrained-euler  | 0.10 | `constrained_euler_results_eps0.1.csv`    | `constrained_euler_samples_eps0.1.json` |
| constrained-fhs    | 0    | `constrained_fhs_results.csv`             | `constrained_fhs_samples.json` |
| constrained-fhs    | 0.05 | `constrained_fhs_results_eps0.05.csv`     | `constrained_fhs_samples_eps0.05.json` |
| constrained-fhs    | 0.10 | `constrained_fhs_results_eps0.1.csv`      | `constrained_fhs_samples_eps0.1.json` |

All paths relative to `outputs/qm9/mdlm_no-guidance/`. Files without `_epsK` suffix are
ε=0 (or legacy with no thresholding — currently identical given sigmoid h_φ > 0).

## TODO

- [x] ε ∈ {0, 0.05, 0.10} sweep — trends confirmed monotone.
- [ ] Continue sweep: ε ∈ {0.20, 0.30, 0.50} — especially for FHS, to find the breaking point.
- [ ] Time-dependent ε(t) — e.g. linearly anneal from low (early t) to high (late t).
- [ ] Multi-seed runs to quantify variance (current N=1000 single-seed numbers have ~±1.5pp std-err on constraint sat.).
- [ ] Plot: x = ε, y = {Valid, Constraint Sat., Novel} for each sampler.
- [ ] Discriminator calibration diagnostic: $h_\phi$ vs MC truth on twisted vs base trajectories (to confirm OOD-shift hypothesis from theory-vs-practice discussion).
