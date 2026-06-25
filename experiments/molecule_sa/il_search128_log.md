# IL+td steps=128 N=500 — Viol@3.0 < 6.5% hyperparameter search

## FINAL SUMMARY (TL;DR)

**Task:** adual IL (Algm2) + time-dep classifier, steps=128, N=500, tune C/ρ/λ₀/λ_max/J(≥2)/n for Viol@3.0<6.5%.

**Best reproducible result (seed1):** `J=4, n=1, C=−log0.95 (0.05129), ρ=0.15, λ₀=2, λ_max=2.9, seed=1`
→ **Viol@3.0 = 4.59%** (valid 392/500). TAG `r9_J4n1_C95_lmax2p9`. Repro:
`bash experiments/molecule_sa/scripts/run_il_search_steps.sh <GPU> 0.05129 0.15 2.0 2.9 4 1 1 r9_J4n1_C95_lmax2p9 128 125`

**Milestones, all found by param search at fixed seed=1 (lowering target each time per request):**
6.5%✅(λmax2.5=6.38%) → 6.0%✅(λmax2.75=5.09%) → 5.0%✅(λmax3.0=4.91%) → best 4.59% (λmax2.9).

**Viol@3.0 progression (J=4, n=1, C=−log0.95, ρ=0.15, λ₀=2, seed=1; only λ_max varied):**
**6.38% (λmax=2.5) → 5.09% (λmax=2.75) → 4.91% (λmax=3.0) → 4.59% (λmax=2.9)**

**Levers learned:** J peaks at 4 (J2/3/4/5 = 7.91/7.16/6.38/7.51 @ λmax2.5); λ_max U-shaped, min≈2.9
(2.25/2.5/2.75/2.9/3.0/3.25/3.75 = 8.01/6.38/5.09/4.59/4.91/6.12/7.0%); λ₀=2 (λ₀≥4 collapses,
12–35%); C clip-dominated (irrelevant); ρ=0.15 (ρ0.20 worse); n=1.

**Robustness caveat (8-seed sweep @ λmax2.9):** mean = **6.49% ± 1.43**, range 4.59–9.36%. The whole
param search was at seed1, which is a favorable draw → the sub-5% headline numbers are seed1-specific
and exactly-reproducible, NOT a robust mean. Cross-seed the config sits ≈ at the 6.5% target. This is
intrinsic to J≥2 IL (averaging kills the useful single-sample stochasticity); matches the prior
steps=64 caveat. **<4.5% appears infeasible** (best draw = seed1 4.59%).

## ALL configs with Viol@3.0 < 6.5% (IL+td, steps=128, N=500)

16 runs cleared the 6.5% bar. **Every one is J=4, n=1, λ₀=2** (the winning regime); they differ only in
λ_max / C / ρ / seed. Sorted by Viol@3.0. (Sample time ≈ 18.3–18.7 min each; n=1 so all are cheap.)

| # | Viol@3.0 | Valid | Unique | Novel&SA≤3.0 | QED | J | n | C | ρ | λ₀ | λ_max | seed | TAG |
| - | -------: | ----: | -----: | -----------: | --: | - | - | - | - | -- | ----- | ---- | --- |
| 1 | **4.59%** ⭐ | 78.4% | 36.4% | 49 | 0.447 | 4 | 1 | −log0.95 | 0.15 | 2 | 2.9  | 1 | r9_J4n1_C95_lmax2p9 |
| 2 | 4.62% | 78.0% | 37.0% | 51 | 0.446 | 4 | 1 | −log0.95 | 0.15 | 2 | 2.85 | 1 | r9_J4n1_C95_lmax2p85 |
| 3 | 4.63% | 77.8% | 36.2% | 48 | 0.444 | 4 | 1 | −log0.95 | 0.15 | 2 | 2.95 | 1 | r9_J4n1_C95_lmax2p95 |
| 4 | 4.65% | 77.4% | 36.0% | 49 | 0.446 | 4 | 1 | −log0.95 | 0.15 | 2 | 3.05 | 1 | r9_J4n1_C95_lmax3p05 |
| 5 | 4.91% | 77.4% | 35.8% | 48 | 0.444 | 4 | 1 | −log0.95 | 0.15 | 2 | 3.0  | 1 | r8_J4n1_C95_lmax3p0 |
| 6 | 5.09% | 78.6% | 38.2% | 45 | 0.453 | 4 | 1 | −log0.95 | 0.15 | 2 | 2.75 | 1 | r5_J4n1_C95_lmax2p75 |
| 7 | 5.41% | 77.6% | 36.0% | 49 | 0.446 | 4 | 1 | −log0.95 | 0.15 | 2 | 3.1  | 1 | r9_J4n1_C95_lmax3p1 |
| 8 | 5.41% | 77.6% | 36.4% | 52 | 0.453 | 4 | 1 | −log0.95 | 0.15 | 2 | 2.9  | 8 | seed_..lmax2p9_s8 |
| 9 | 5.60% | 78.6% | 35.4% | 45 | 0.462 | 4 | 1 | −log0.95 | 0.15 | 2 | 2.9  | 5 | seed_..lmax2p9_s5 |
| 10 | 6.08% | 79.0% | 33.2% | 43 | 0.446 | 4 | 1 | −log0.95 | 0.15 | 2 | 3.5  | 1 | r8_J4n1_C95_lmax3p5 |
| 11 | 6.12% | 78.4% | 33.6% | 47 | 0.454 | 4 | 1 | −log0.95 | 0.15 | 2 | 3.25 | 1 | r8_J4n1_C95_lmax3p25 |
| 12 | 6.14% | 78.2% | 42.8% | 51 | 0.460 | 4 | 1 | −log0.90 | 0.15 | 2 | 2.5  | 1 | r7_J4n1_C90_lmax2p5_r15 |
| 13 | 6.31% | 79.2% | 33.0% | 48 | 0.461 | 4 | 1 | −log0.95 | 0.15 | 2 | 2.9  | 4 | seed_..lmax2p9_s4 |
| 14 | 6.38% | 78.4% | 42.8% | 51 | 0.460 | 4 | 1 | −log0.95 | 0.15 | 2 | 2.5  | 1 | r5_J4n1_C95_lmax2p5 |
| 15 | 6.38% | 78.4% | 42.8% | 51 | 0.460 | 4 | 1 | −log0.99 | 0.15 | 2 | 2.5  | 1 | r7_J4n1_C99_lmax2p5_r15 |
| 16 | 6.39% | 78.2% | 42.6% | 51 | 0.460 | 4 | 1 | −log0.90 | 0.20 | 2 | 2.5  | 1 | r5_J4n1_C90_lmax2p5 |

Rows 8/9/13 are the λmax2.9 winner at seeds 8/5/4 (the other seeds 2/3/6/7 = 6.65–9.36% missed the bar).
Rows 14/15 are bit-identical (C=−log0.95 vs −log0.99) → C is clip-dominated. Artifacts:
`outputs/qm9/il_search128/<TAG>/{results.csv,samples.json,*traj.json}`.

---


**Goal:** adual **IL (Algm 2) + time-dep classifier**, `steps=128`, `N=500`, tune
`C, ρ, λ₀, λ_max, J(≥2), n` to push **Viol@3.0 < 6.5%**. GPU 7 only.

**Fixed setup:** diffusion ckpt `outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt`,
time-dep classifier `outputs/qm9/classifier/sa_score_le_3.0_absorbing_state_T-0_timedep/checkpoints/best.ckpt`,
`guidance.gamma_adual_inner_loop=True`, `gamma_schedule=adaptive_dual`,
`classifier_time_conditioning=True`, batch=4, num_sample_batches=125.

**Launcher:** `experiments/molecule_sa/scripts/run_il_search_steps.sh GPU C RHO L0 LMAX J N SEED TAG STEPS [NBATCH]`
Outputs: `outputs/qm9/il_search128/<TAG>/results.csv` (col `viol_tau_3.0`).

**Prior context (steps=128 IL, from sa_dcbg_adual_samplefirst_timedep.md):**
- best IL so far: J=4,n=4,C=−log0.90,ρ=0.2,λ₀=2,λ_max=5 → **7.55%**
- J=4,n=4,C=−log0.99,ρ=0.2,λ₀=2,λ_max=5 → 7.94%; λ_max=10 → 8.47%
- J=1,n=1,C=−log0.99,ρ=0.2,λ₀=2,λ_max=5 → 8.11%
- sf best (J=1) → 7.32%. **Low λ_max never tried with J≥2 at steps=128** → main lever.
- steps=64 J≥2 winner: J=2,n=1,C=−log0.95,ρ=0.15,λ₀=2,λ_max=2.5,seed=1 → 7.26%.

C values: −log0.99=0.01005, −log0.95=0.05129, −log0.90=0.10536, −log0.8=0.2231.

**Run policy:** SEQUENTIAL on GPU7 (single run already ~80% util → concurrency gives no
throughput gain; 3-concurrent attempt got SIGKILL'd mid-run, likely GPU mem spike). seed=1
fixed; explore params broadly first, only tune seed as last resort (≤8 seeds/config then move on).

## ✅ TARGET <6.5% ACHIEVED (seed=1, no seed-tuning needed)

**Config:** `J=4, n=1, C=−log0.95 (0.05129), ρ=0.15, λ₀=2, λ_max=2.5, seed=1`, IL+td, steps=128, N=500.
**Result: Viol@3.0 = 6.38%** (valid 78.4%, 392/500). TAG `r5_J4n1_C95_lmax2p5`.
Reproduce: `bash experiments/molecule_sa/scripts/run_il_search_steps.sh <GPU> 0.05129 0.15 2.0 2.5 4 1 1 r5_J4n1_C95_lmax2p5 128 125`
Artifacts: `outputs/qm9/il_search128/r5_J4n1_C95_lmax2p5/{results.csv,samples.json}`.
Found purely by parameter search (J lever: J2→J3→J4 at λmax2.5 = 7.91→7.16→6.38%); seed fixed at 1.

**→ New target: Viol@3.0 < 6.0%.** Continue: push J=5/6 + λmax fine grid in region A.

## ✅✅ TARGET <6.0% ACHIEVED (seed=1, no seed-tuning)

**Config:** `J=4, n=1, C=−log0.95 (0.05129), ρ=0.15, λ₀=2, λ_max=2.75, seed=1`, IL+td, steps=128, N=500.
**Result: Viol@3.0 = 5.09%** (valid 78.6%, 393/500). TAG `r5_J4n1_C95_lmax2p75`.
Reproduce: `bash experiments/molecule_sa/scripts/run_il_search_steps.sh <GPU> 0.05129 0.15 2.0 2.75 4 1 1 r5_J4n1_C95_lmax2p75 128 125`
Artifacts: `outputs/qm9/il_search128/r5_J4n1_C95_lmax2p75/`.
**λmax is the decisive lever at J=4: λmax 2.25/2.5/2.75 = 8.01/6.38/5.09% — still dropping at 2.75.**
**→ Next target: <5.0%. Push λmax to 3.0/3.25/3.5 at J=4.**

## ✅✅✅ TARGET <5.0% ACHIEVED (seed=1, no seed-tuning)

**Config:** `J=4, n=1, C=−log0.95 (0.05129), ρ=0.15, λ₀=2, λ_max=3.0, seed=1`, IL+td, steps=128, N=500.
**Result: Viol@3.0 = 4.91%** (valid 77.4%, 387/500). TAG `r8_J4n1_C95_lmax3p0`.
Reproduce: `bash experiments/molecule_sa/scripts/run_il_search_steps.sh <GPU> 0.05129 0.15 2.0 3.0 4 1 1 r8_J4n1_C95_lmax3p0 128 125`
λmax curve flattening (2.75→3.0 = 5.09→4.91%); min near λmax 3.0–3.75 (sweeping). **→ Next target <4.5%.**

## Results

| TAG | J | n | C | ρ | λ₀ | λ_max | seed | Valid | Viol@3.0 | time | notes |
| --- | - | - | - | - | -- | ----- | ---- | ----- | -------- | ---- | ----- |
| b1_lmax2p5_C95_r15 | 2 | 1 | 0.05129 | 0.15 | 2 | 2.5 | 1 | 78.4% | **7.91%** | 16.5m | low-λmax basin ~7.9% |

**Reflection (after b1):** low-λ_max (2-3.5) basin sits ~7.5-8% at steps=128 (vs 7.26% at
steps=64 — finer grid weakens the low-λmax trick). Pivot to **λ₀ lever**: steps=256 sf winner
used λ₀=4,λ_max=10 → 6.05%. Batch 2 explores λ₀=3-4 with larger λ_max under IL/J=2.

### Batch 2 — λ₀ lever (queue, seed=1)

| TAG | J | n | C | ρ | λ₀ | λ_max | seed | Valid | Viol@3.0 | notes |
| --- | - | - | - | - | -- | ----- | ---- | ----- | -------- | ----- |
| b2_l04_lmax10_C99_r20_J2 | 2 | 1 | 0.01005 | 0.2 | 4 | 10 | 1 | 57% | 13.99% | λ₀=4 BAD |
| b2_l04_lmax8_C99_r20_J2  | 2 | 1 | 0.01005 | 0.2 | 4 | 8  | 1 | 56% | 13.83% | λ₀=4 BAD |
| b2_l04_lmax5_C99_r20_J2  | 2 | 1 | 0.01005 | 0.2 | 4 | 5  | 1 | 56% | 12.41% | λ₀=4 BAD |
| b2_l04_lmax10_C99_r10_J2 | 2 | 1 | 0.01005 | 0.1 | 4 | 10 | 1 | 61% | 12.83% | λ₀=4 BAD |
| g2_l04_lmax10_C80_r20_J2 | 2 | 1 | 0.22314 | 0.2 | 4 | 10 | 1 | 58% | 13.01% | λ₀=4 BAD |
| g2_l06_lmax10_C99_r20_J2 | 2 | 1 | 0.01005 | 0.2 | 6 | 10 | 1 | 4%  | 35.0%  | λ₀=6 COLLAPSE |

**Reflection (after b2): λ₀ lever is WRONG for IL.** High λ₀ (=4/6) over-constrains: IL's
J≥2 inner loop pins λ high → over-trusts imperfect classifier → SA violated (viol ↑ to 12-14%)
AND validity collapses (78%→57%, λ₀=6→4%). The sf/steps=256 λ₀=4 trick does NOT transfer to IL.
**Good regime = low λ₀(=2) + low/moderate λ_max.** Killed all remaining λ₀≥4 configs.
Best so far: b1 (λ₀=2,λmax2.5,J2,n1 → 7.91%) and doc J4n4/C90/λmax5 → 7.55%.

### Batch 3 — dense sweep of GOOD regime (λ₀=2 fixed), seed=1
GPU7 = region B (doc-winner: C90, λmax~5, n=4, J=2/3/4); GPU2 = region A (low λmax, n=1/8, J=2/3).

| TAG | J | n | C | ρ | λ₀ | λ_max | seed | Valid | Viol@3.0 | notes |
| --- | - | - | - | - | -- | ----- | ---- | ----- | -------- | ----- |
| b1 (ref)            | 2 | 1 | 0.05129 | 0.15 | 2 | 2.5 | 1 | 78.4% | 7.91% | region-A best |
| b3_J2n1_C95_lmax3_r15 | 2 | 1 | 0.05129 | 0.15 | 2 | 3.0 | 1 | 78.8% | 8.63% | λmax↑ worse → go <2.5 |
| b3_J2n1_C90_lmax3_r20 | 2 | 1 | 0.10536 | 0.20 | 2 | 3.0 | 1 | 77.8% | 7.97% | C90>C95 at λmax3 |
| **b3_J4n4_C90_lmax5_l02** | 4 | 4 | 0.10536 | 0.20 | 2 | 5.0 | 1 | 76.8% | **7.55%** ⭐ | reproduces doc best; global best so far |
| b3_J3n4_C90_lmax5_l02 | 3 | 4 | 0.10536 | 0.20 | 2 | 5.0 | 1 | 84.0% | 9.05% | J=3<J=4 (more J→stronger) |
| b3_J2n4_C90_lmax3_r20 | 2 | 4 | 0.10536 | 0.20 | 2 | 3.0 | 1 | 81.6% | 8.09% | |

**J is the key lever at steps=128: J4n4=7.55% < J3n4=9.05%. Batch 4 → push J=5/6 + looser C
(−log0.85/0.80) at winning point.** (Note: contradicts steps=64 "J2 best" — finer grid favors deeper solve.)

| b3_J2n4_C90_lmax5_l02 | 2 | 4 | 0.10536 | 0.20 | 2 | 5.0 | 1 | 83.4% | 9.11% | |
| **b3_J3n1_C95_lmax2p5_r15** | 3 | 1 | 0.05129 | 0.15 | 2 | 2.5 | 1 | 81.0% | **7.16%** ⭐⭐ | NEW BEST: region A + J=3 + n=1 |

**KEY (after batch 3 full): NEW GLOBAL BEST = J3n1/C95/λmax2.5/ρ0.15 = 7.16%** (region A, cheap n=1).
J2→J3 at λmax2.5: 7.91%→7.16%. Two winning directions: (A) low-λmax n=1 high-J [cheap, best],
(B) J4n4 C90 λmax5 [7.55%]. **Batch 5 (GPU7) = region A J=4/5 + λmax fine grid {2.25,2.5,2.75}.**

### Batch 7 — fine-tune around J4 optimum (toward <6.0%)
- **r7_J4n1_C99_lmax2p5_r15 → 6.38%, valid 392 — BIT-IDENTICAL to C95.** ⇒ **C is clip-dominated
  (λ pinned at λmax most of trajectory); C is NOT a useful lever here.** Remaining levers = ρ, λmax.
- r5_J3n1_C95_lmax2p25 → 7.88% (J3+λmax2.25 both suboptimal).

### λmax curve @ J=4 (C95, ρ0.15, λ₀2, n=1, seed=1) — the decisive lever
| λmax | 2.25 | 2.5 | 2.75 | 3.0+ |
| ---- | ---- | --- | ---- | ---- |
| Viol@3.0 | 8.01% | 6.38% | **5.09%** | 3.0=4.91, 3.75=7.0 (U-shape, min≈3.0) |

**λmax is U-shaped, minimum ≈2.9.** Fine grid: λmax 2.75/2.9/3.0/3.25 = 5.09/**4.59**/4.91/6.12%.
**Current best = λmax2.9 → Viol 4.59%** (valid 392). Fine grid: 2.9=4.59, 2.95=4.63, 3.0=4.91%.
Param floor ≈4.59% (plateau 2.9–3.05 = 4.59/4.63/4.91/4.65, noisy). **ρ0.20@λmax3.0 = 6.54% > ρ0.15
→ ρ=0.15 near-optimal.** All levers now characterized: J=4, λmax≈2.9, C irrelevant, ρ=0.15, λ₀=2, n=1.
**Params exhausted → seed sweep on best config (λmax2.9) seeds 2–8 (user-authorized last resort) for <4.5% + robustness.**

Full λmax curve @ J4/C95/ρ0.15/λ₀2/n1/seed1: 2.85=?, 2.9=**4.59**, 2.95=4.63, 3.0=4.91, 3.05=4.65,
3.1=5.41, 3.25=6.12, 3.5=6.08, 3.75=7.0 → clean U, min λmax≈2.9. (J5@3.25=6.87 confirms J=4 optimal.)

### Seed sweep @ J4n1/C95/ρ0.15/λ₀2/λmax2.9 (seed1=4.59%)
| seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| ---- | - | - | - | - | - | - | - | - |
| Viol@3.0 | 4.59% | 6.88% | 7.11% | 6.31% | 5.60% | 6.65% | 9.36% | 5.41% |

**8-seed stats @ λmax2.9: mean = 6.49% ± 1.43 (sample std), range 4.59–9.36%, median ~6.5%.**

**Seed variance is VERY LARGE (J≥2): range 4.59%–9.36%, ~5pt; first-6 mean ≈ 6.82%.** seed1=4.59% is
a lucky low outlier, seed7=9.36% a high one. **The entire param sweep was at seed1 → λmax2.9 is
seed1-overfit.** Honest picture:
- The headline 6.38/5.09/4.91/4.59% are **seed1-only**, exactly-reproducible (config, seed1), NOT robust.
- Cross-seed this config averages ~6.8% (high variance); even <6.5% is not met per-seed by every draw.
- **<4.5% infeasible** (best draw seed1=4.59%). Per user rule (>8 seeds not meeting target → switch),
  but params are already the seed1-optimum; the limiting factor is intrinsic J≥2 seed variance.
(λmax2.85/seed1 = 4.62% confirms the ~4.6% plateau at the U-minimum.)
(C90 @ λmax2.5 = 6.39% ≈ C95 6.38% → C confirmed irrelevant. ρ/C weak; **λmax dominant**.)

### Batch 5 — region A high-J around 7.16% winner (GPU7, n=1, seed=1)

| TAG | J | n | C | ρ | λ₀ | λ_max | seed | Valid | Viol@3.0 | notes |
| --- | - | - | - | - | -- | ----- | ---- | ----- | -------- | ----- |
| **r5_J4n1_C95_lmax2p5**  | 4 | 1 | 0.05129 | 0.15 | 2 | 2.5  | 1 | 78.4% | **6.38%** ✅ | <6.5% ACHIEVED |
| r5_J4n1_C95_lmax2p25 | 4 | 1 | 0.05129 | 0.15 | 2 | 2.25 | 1 | 77.4% | 8.01% | λmax<2.5 worse → 2.5 is sweet spot |
| r5_J5n1_C95_lmax2p5  | 5 | 1 | 0.05129 | 0.15 | 2 | 2.5  | 1 | 77.2% | 7.51% | J=5 WORSE → J peaks at 4 |

**KEY: J peaks at J=4. J2→J3→J4→J5 @ λmax2.5 = 7.91/7.16/6.38/7.51.** λmax peaks at 2.5
(2.25=8.01). Path to <6.0% = fine-tune C/ρ/λmax AROUND J4n1/C95/λmax2.5/ρ0.15.
Batch 7 (GPU2) = C∈{99,90,85}, ρ∈{0.10,0.20}, λmax2.6 at J=4. Batch 5 (GPU7) tail still
has J4 λmax2.75 + J4 C90.
| r5_J3n1_C95_lmax2p25 | 3 | 1 | 0.05129 | 0.15 | 2 | 2.25 | 1 | ? | ? | |
| r5_J4n1_C95_lmax2p75 | 4 | 1 | 0.05129 | 0.15 | 2 | 2.75 | 1 | ? | ? | |
| r5_J4n1_C90_lmax2p5  | 4 | 1 | 0.10536 | 0.20 | 2 | 2.5  | 1 | ? | ? | C90 variant |

**Reflection (batch 3 partial):** region B (J4n4/C90/λmax5) = 7.55% confirmed as global best;
region A (J2n1 low λmax) stuck 7.9-8.6%. C90>C95. **Next lever: looser C (−log0.85/0.80) at the
J4n4 winning point — never tried; lets λ decay after classifier clears its (higher) −C target,
reducing over-constraint.** Batch 4 = C-looser + λmax fine grid {4,4.5,5} around J4n4/λ₀2.
