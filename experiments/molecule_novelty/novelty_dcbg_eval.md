# Novelty-Constrained Evaluation — MDLM + D-CBG / Adaptive-Dual

Beating [Cardei et al. NeurIPS 2025, *Constrained Discrete Diffusion* (arXiv:2503.09790)](https://arxiv.org/abs/2503.09790) Fig 4 **RIGHT** (Novelty constraint, "No Guidance" block).

**Date:** 2026-06-14.
**Setup:** QM9 SMILES, MDLM (absorbing-state diffusion, parameterization=subs, T=0, model=small/dit, length=32), `sampling.steps=128`, `seed=1`, **N=500**.
**Constraint:** generated molecule must be **novel** (canonical SMILES ∉ QM9 train). The guidance classifier predicts **p(novel | x_t)** directly (class 1 = novel); D-CBG conditions on class 1, adaptive-dual presses the constraint via a per-step Lagrangian dual variable λ.
**Headline KPI:** number of **valid & novel** molecules (maximize). `Viol %` = fraction of *valid* generations that are **not** novel = `1 − novel_rate` (CDD's "Viol (%)" column).

---

## Pipeline (5 stages)

1. **Novelty labels** — QM9-train is all label-0 ("not novel"), so the training set is built offline (`build_novelty_dataset.py`): generate ~50 k valid SMILES from the frozen base MDLM, label `novel=1` if canonical ∉ QM9-train else `0`, mix in QM9-train as guaranteed label-0. Result: 50,191 rows, novel 21,185 / not-novel 29,006 (≈42 % novel, naturally balanced — no extra mixing needed).
2. **`qm9_novel` dataloader branch** + `configs/data/qm9_novel.yaml` (`label_col=novel`, num_classes=2).
3. **Train D-CBG classifier** `p(novel | x_t)` — tiny-classifier DIT, 25 000 steps. Best `val/cross_entropy = 0.509` (final-epoch precision/recall = 0.76/0.61; best.ckpt = early generalizing point, overfits after).
4. **Sample** MDLM + D-CBG (`guidance.condition=1`), constant-γ baselines and adaptive-dual sweep.
5. **Eval** (`novelty_eval.py`) — Valid / Unique / Valid&Novel / Viol / QED + adaptive-dual λ trajectory dump.

---

## Combined results — CBG vs adaptive-dual (N=500, seed=1, steps=128)

All runs share N=500 / seed=1 / steps=128 / the same `p(novel|x_t)` classifier.
`Viol %` = (# valid not-novel)/(# valid) = 1 − novel_rate. **Valid & Novel** is the headline KPI.

| Method | Config | Valid | novel_rate | Viol % | **Valid & Novel** | QED |
| :----- | :----- | ----: | ---------: | -----: | ----------------: | --: |
| no-guidance      | —              |   311 |   32.5 % | 67.5 % |               101 | 0.458 |
| **CBG** (D-CBG)  | γ = 1          |   257 |   62.6 % | 37.4 % |               161 | 0.450 |
| CBG              | γ = 2          |   219 |   80.4 % | 19.6 % |          176 ◦ |       0.449 |
| CBG              | γ = 3          |   169 |   89.3 % | 10.7 % |               151 | 0.458 |
| CBG              | γ = 4          |   117 |   92.3 % |  7.7 % |               108 | 0.462 |
| CBG              | γ = 5          |    66 |   97.0 % |  3.0 % |                64 | 0.474 |
| CBG              | γ = 6          |    33 |  100.0 % |  0.0 % |                33 | 0.525 |
| CBG              | γ = 7          |    20 |  100.0 % |  0.0 % |                20 | 0.505 |
| CBG              | γ = 8          |     9 |  100.0 % |  0.0 % |                 9 | 0.479 |
| CBG              | γ = 9          |     9 |  100.0 % |  0.0 % |                 9 | 0.409 |
| CBG              | γ = 10         |     2 |  100.0 % |  0.0 % |                 2 | 0.356 |
| **adaptive-dual** | C=-log(0.8) ρ=0.5 λmax=2  |   247 |   77.3 % | 22.7 % |               191 | 0.440 |
| adaptive-dual    | C=-log(0.8) ρ=0.5 λmax=3  |   245 |   82.4 % | 17.6 % |               202 | 0.448 |
| adaptive-dual    | C=-log(0.8) ρ=0.5 λmax=5  |   235 |   86.8 % | 13.2 % |               204 | 0.439 |
| adaptive-dual    | C=-log(0.8) ρ=0.5 λmax=10 |   231 |   90.9 % |  9.1 % |               210 | 0.430 |
| adaptive-dual    | C=-log(0.8) ρ=0.5 λmax=20 |   231 |   90.9 % |  9.1 % |               210 | 0.429 |
| adaptive-dual    | C=-log(0.8) ρ=0.5 λmax=50 |   231 |   90.9 % |  9.1 % |               210 | 0.429 |
| adaptive-dual    | C=-log(0.99) ρ=0.5 λmax=2  |   234 |   77.8 % | 22.2 % |               182 | 0.447 |
| adaptive-dual    | C=-log(0.99) ρ=0.5 λmax=3  |   222 |   84.7 % | 15.3 % |               188 | 0.450 |
| adaptive-dual    | C=-log(0.99) ρ=0.5 λmax=5  |   210 |   89.1 % | 11.0 % |               187 | 0.438 |
| adaptive-dual    | C=-log(0.99) ρ=0.5 λmax=10 |   188 |   94.2 % |  5.9 % ✗ |             177 | 0.430 |
| adaptive-dual    | C=-log(0.99) ρ=0.2 λmax=2  |   254 |   73.2 % | 26.8 % |               186 | 0.437 |
| adaptive-dual    | C=-log(0.99) ρ=0.2 λmax=3  |   259 |   80.7 % | 19.3 % |               209 | 0.443 |
| adaptive-dual    | C=-log(0.99) ρ=0.2 λmax=5  |   249 |   86.8 % | 13.3 % |               216 | 0.434 |
| adaptive-dual    | C=-log(0.99) ρ=0.2 λmax=10 |   246 |   91.1 % |  **8.9 %** |           224 ⭐ | 0.429 |

◦ best CBG point (γ=2).  ⭐ **best operating point (viol-priority): lowest Viol that keeps validity high** — C=−log(0.99) ρ=0.2 λmax=10 → **Viol 8.9 %, valid 246, V&N 224**.  ✗ lower Viol (5.9 %) but validity craters to 188 — not worth it.

**Viol-centric win.** To reach Viol ≈ 9 %, CBG needs γ≈4 (Viol 7.7 %, valid 117). adual hits **Viol 8.9 % while keeping valid 246** — >2× the valid molecules at matched Viol.

---

## Combined results — steps=256 (CBG + adaptive-dual, N=500, seed=1)

| Method | Config | Valid | novel_rate | Viol % | **Valid & Novel** | QED |
| :----- | :----- | ----: | ---------: | -----: | ----------------: | --: |
| no-guidance      | γ = 0          |   321 |   35.8 % | 64.2 % |               115 | 0.449 |
| **CBG**          | γ = 1          |   262 |   60.7 % | 39.3 % |               159 | 0.450 |
| CBG              | γ = 2          |   233 |   79.0 % | 21.0 % |          184 ◦ |       0.460 |
| CBG              | γ = 3          |   196 |   86.7 % | 13.3 % |               170 | 0.469 |
| CBG              | γ = 4          |   146 |   91.8 % |  8.2 % |               134 | 0.467 |
| CBG              | γ = 5          |   115 |   95.7 % |  4.4 % |               110 | 0.467 |
| CBG              | γ = 6          |    73 |   97.3 % |  2.7 % |                71 | 0.476 |
| CBG              | γ = 7          |    28 |  100.0 % |  0.0 % |                28 | 0.480 |
| CBG              | γ = 8          |     5 |  100.0 % |  0.0 % |                 5 | 0.573 |
| CBG              | γ = 9          |     8 |  100.0 % |  0.0 % |                 8 | 0.436 |
| CBG              | γ = 10         |     2 |  100.0 % |  0.0 % |                 2 | 0.503 |
| **adaptive-dual** | C=−log(0.99) ρ=0.5 λmax=5  |   205 |   90.2 % |  9.8 % |               185 | 0.445 |
| adaptive-dual    | C=−log(0.99) ρ=0.5 λmax=10 |   189 |   97.9 % |  2.1 % |               185 | 0.439 |
| adaptive-dual    | C=−log(0.99) ρ=0.1 λmax=5  |   248 |   84.3 % | 15.7 % |               209 | 0.441 |
| adaptive-dual    | C=−log(0.99) ρ=0.1 λmax=10 |   244 |   88.9 % | 11.1 % |               217 | 0.441 |
| adaptive-dual    | C=−log(0.8)  ρ=0.5 λmax=5  |   205 |   89.3 % | 10.7 % |               183 | 0.445 |
| adaptive-dual    | C=−log(0.8)  ρ=0.5 λmax=10 |   204 |   93.1 % |  **6.9 %** |           190 ⭐ | 0.435 |
| adaptive-dual    | C=−log(0.8)  ρ=0.1 λmax=5  |   252 |   86.9 % | 13.1 % |               219 | 0.437 |
| adaptive-dual    | C=−log(0.8)  ρ=0.1 λmax=10 |   254 |   89.4 % | 10.6 % |               227 ✦ | 0.432 |

◦ best CBG point (γ=2).  ⭐ **best operating point (viol-priority): lowest Viol that keeps validity high** — C=−log(0.8) ρ=0.5 λmax=10 → **Viol 6.9 %, valid 204, V&N 190**.  ✦ max Valid&Novel (227, but Viol 10.6 %).

**Viol-centric win.** To reach Viol ≈ 7 %, CBG needs γ≈4 (valid 146) or γ=5 (Viol 4.4 %, valid 115) — validity already cratering. adual hits **Viol 6.9 % while keeping valid 204** (+40 % validity over CBG-γ=4 at matched Viol). The even-lower-Viol point C=−log(0.99) ρ=0.5 λmax=10 (Viol 2.1 %) costs too much validity (189) to be worth it.

---

## Combined results — steps=512 (CBG + adaptive-dual, N=500, seed=1)

| Method | Config | Valid | novel_rate | Viol % | **Valid & Novel** | QED |
| :----- | :----- | ----: | ---------: | -----: | ----------------: | --: |
| no-guidance      | γ = 0          |   319 |   32.0 % | 68.0 % |               102 | 0.447 |
| **CBG**          | γ = 1          |   274 |   64.2 % | 35.8 % |          176 ◦ |       0.452 |
| CBG              | γ = 2          |   198 |   77.3 % | 22.7 % |               153 | 0.456 |
| CBG              | γ = 3          |   164 |   86.0 % | 14.0 % |               141 | 0.469 |
| CBG              | γ = 4          |   148 |   93.2 % |  6.8 % |               138 | 0.460 |
| CBG              | γ = 5          |   127 |   96.9 % |  3.2 % |               123 | 0.459 |
| CBG              | γ = 6          |    97 |   97.9 % |  2.1 % |                95 | 0.487 |
| CBG              | γ = 7          |    64 |  100.0 % |  0.0 % |                64 | 0.486 |
| CBG              | γ = 8          |    18 |  100.0 % |  0.0 % |                18 | 0.482 |
| CBG              | γ = 9          |    11 |  100.0 % |  0.0 % |                11 | 0.544 |
| CBG              | γ = 10         |     3 |  100.0 % |  0.0 % |                 3 | 0.468 |
| **adaptive-dual** | C=−log(0.99) ρ=0.5 λmax=5  |   165 |   92.1 % |  7.9 % |               152 | 0.437 |
| adaptive-dual    | C=−log(0.99) ρ=0.5 λmax=10 |   162 |   92.0 % |  8.0 % |               149 | 0.446 |
| adaptive-dual    | C=−log(0.99) ρ=0.1 λmax=5  |   214 |   87.4 % | 12.6 % |               187 | 0.438 |
| adaptive-dual    | C=−log(0.99) ρ=0.1 λmax=10 |   224 |   93.3 % |  **6.7 %** |           209 ⭐ | 0.435 |
| adaptive-dual    | C=−log(0.8)  ρ=0.5 λmax=5  |   179 |   89.9 % | 10.1 % |               161 | 0.446 |
| adaptive-dual    | C=−log(0.8)  ρ=0.5 λmax=10 |   174 |   89.1 % | 10.9 % |               155 | 0.446 |
| adaptive-dual    | C=−log(0.8)  ρ=0.1 λmax=5  |   229 |   88.2 % | 11.8 % |               202 | 0.433 |
| adaptive-dual    | C=−log(0.8)  ρ=0.1 λmax=10 |   246 |   89.0 % | 11.0 % |               219 | 0.432 |
| adaptive-dual    | C=−log(0.5)  ρ=0.5 λmax=5  |   253 |   84.6 % | 15.4 % |               214 | 0.435 |
| adaptive-dual    | C=−log(0.5)  ρ=0.5 λmax=10 |   254 |   87.4 % | 12.6 % |               222 | 0.426 |
| adaptive-dual    | C=−log(0.5)  ρ=0.1 λmax=5  |   270 |   85.2 % | 14.8 % |               230 | 0.434 |
| adaptive-dual    | C=−log(0.5)  ρ=0.1 λmax=10 |   280 |   88.2 % | 11.8 % |               247 ✦ | 0.430 |

◦ best CBG point (γ=1).  ⭐ **best operating point (viol-priority)**: C=−log(0.99) ρ=0.1 λmax=10 → **Viol 6.7 %, valid 224, V&N 209** (lowest Viol *and* high validity).  ✦ max Valid&Novel (247, C=−log(0.5) ρ=0.1 λmax=10, but Viol 11.8 %).
**Viol-centric win.** To reach Viol ≈ 7 %, CBG needs γ=4 (valid 148); adual hits **Viol 6.7 % at valid 224** (+51 % validity at matched Viol). Gentle ρ=0.1 preserves validity; ρ=0.5 collapses it (162–179). **The budget C is the viol/validity dial**: strict C=−log(0.99) → lowest Viol (6.7 %, valid 224); loose C=−log(0.5) → highest validity & V&N (280 / 247) at higher Viol (~12 %).

---

## Results — steps=1024 (CBG, N=500, seed=1)

Constant-γ D-CBG at `sampling.steps=1024` (γ=0 = no-guidance). CBG only (adual not run at 1024).

| Method | Config | Valid | novel_rate | Viol % | **Valid & Novel** | QED |
| :----- | :----- | ----: | ---------: | -----: | ----------------: | --: |
| no-guidance | γ = 0  |   331 |   37.2 % | 62.8 % |               123 | 0.452 |
| CBG         | γ = 1  |   285 |   62.5 % | 37.5 % |               178 | 0.448 |
| CBG         | γ = 2  |   243 |   81.1 % | 18.9 % |          **197** ◦ | 0.457 |
| CBG         | γ = 3  |   180 |   87.2 % | 12.8 % |               157 | 0.452 |
| CBG         | γ = 4  |   160 |   91.9 % |  8.1 % |               147 | 0.450 |
| CBG         | γ = 5  |   130 |   96.9 % |  3.1 % |               126 | 0.448 |
| CBG         | γ = 6  |   121 |   95.9 % |  4.1 % |               116 | 0.455 |
| CBG         | γ = 7  |    93 |   92.5 % |  7.5 % |                86 | 0.461 |
| CBG         | γ = 8  |    49 |   95.9 % |  4.1 % |                47 | 0.484 |
| CBG         | γ = 9  |    21 |  100.0 % |  0.0 % |                21 | 0.466 |
| CBG         | γ = 10 |    12 |  100.0 % |  0.0 % |                12 | 0.508 |

◦ best CBG point (γ=2 = 197). Same trade-off (validity collapses as γ grows; high-γ Viol noisy at N=500).

---

## Files

- Data: `build_novelty_dataset.py` → `.data_cache/qm9_novelty_scored`
- Classifier: `outputs/qm9/classifier/novelty_absorbing_state_T-0/checkpoints/best.ckpt`
- Eval: `novelty_eval.py` · Scripts: `scripts/{build_dataset,train_novelty_classifier,sample_novelty_baselines,sample_novelty_dcbg_adaptive}.sh`
- Results CSVs: `results/{noguidance,dcbg_gamma{1..10}}_results.csv`
