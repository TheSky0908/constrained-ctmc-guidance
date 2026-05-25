# Idea Discovery Report — CTMC Algorithms for Hard-Constraint Discrete Diffusion

**Direction:** New CTMC-based algorithms to raise hard-constraint satisfaction rate in discrete diffusion sampling, building on the existing h-twist samplers (`constrained_{ddpm,euler,fhs}.py`).
**Date:** 2026-05-22
**Pipeline:** focused literature survey → brainstorm → novelty triage (full `/novelty-check`, `/research-review`, `/research-refine-pipeline` deferred until you pick a top idea).

---

## Executive Summary

Current best is `constrained-fhs` at **45.1% constraint satisfaction** (ε=0.10), bottlenecked at **65.3% validity × 69% novel-of-valid**. The h-twist is the optimal one-step proposal under Doob; the gap is *not* the twist itself but the **sampling apparatus around it**: no resampling, no backtracking, off-policy h<sub>φ</sub>, one-step Bellman, no exploitation of constraint structure.

Eight ideas below, organised by where they attack the loss. **Recommended pair: Idea 1 (Twisted-SMC for hard constraints with ESS resampling) + Idea 3 (h-aware ReMask)** — orthogonal mechanisms, both implementable in <2 days, expected combined lift to **65–80% constraint sat**.

---

## Current Bottleneck Decomposition

| Source of loss                                  | Current value     | Root cause                                                                 |
| ----------------------------------------------- | ----------------- | -------------------------------------------------------------------------- |
| Validity (SMILES syntax + valence)              | 65.3% (FHS,ε=.10) | p<sub>θ</sub> is the only validity signal; h<sub>φ</sub> is novelty-only.  |
| Novel-of-valid                                  | 69%               | h<sub>φ</sub> is one-step myopic + OOD under twist.                        |
| ε-threshold abort risk                          | rises with ε      | local greedy pruning, no global recovery when all candidates fall below ε. |
| Discriminator (h<sub>φ</sub>) calibration drift | unmeasured        | h<sub>φ</sub> trained on **unconditional** FHS trajectories, deployed on **twisted** ones — distributional shift.|

Every idea below targets one (or several) of these rows.

---

## Recent Landscape (what's already done — to avoid duplicating)

| Year     | Paper                                                                  | Mechanism                                                            | Why it differs from our setup                                        |
| -------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 2024     | **SVDD** ([2408.08252](https://arxiv.org/abs/2408.08252))              | Soft value-based decoding for both continuous & discrete; expects a reward signal. | Soft reward, no hard 1{X<sub>0</sub>∈C}; no SMC; no remask.    |
| 2024     | **TPPF** ([2409.02399](https://arxiv.org/abs/2409.02399))              | Twisted particle filter, neural twist via KL-divergence training; continuous-time view. | Continuous state, not discrete CTMC.                                 |
| 2025 ICLR| **Test-Time Alignment / SMC-DDM** ([2505.22524](https://arxiv.org/abs/2505.22524)) | Twisted SMC for discrete diffusion, Taylor-expansion proposal + Gumbel-Softmax. | Soft reward + small particle counts; *not* tested with hard 1-indicator and not on QM9 novelty.|
| 2025     | **CDM** ([cdm-smc.github.io](https://cdm-smc.github.io/))              | Contrastive amortised twist function (positive/negative samples).    | Reward-tilted target; does **not** discuss resampling, hard constraints, or molecules.|
| 2025     | **ReMDM** ([2503.00307](https://arxiv.org/abs/2503.00307))             | Remasking discrete diffusion, inference-time scaling.                | No explicit guidance / h-transform — just iterative refinement.      |
| 2025     | **Constrained Discrete Diffusion** ([2503.09790](https://arxiv.org/abs/2503.09790)) | Training-free constraint imposition; tasks: toxicity / molecules / instruction-following. | Constraints are projection-based, not CTMC-twist; QM9-novelty not tested. |
| 2026     | **Stratified Hazard Sampling** ([2601.02799](https://arxiv.org/html/2601.02799)) | Min-variance event scheduling for unconditional CTMC discrete diffusion. | Unconditional; orthogonal to h-twist, **combinable**.                |
| 2026     | **Sample-Efficient Conditionals** ([2602.20293](https://arxiv.org/pdf/2602.20293)) | Efficient conditional estimators.                                    | Different conditioning target.                                       |

**Conclusion:** twisted-SMC for *reward-tilted* discrete diffusion is solidly explored. Twisted-SMC for *hard-constraint* discrete CTMC with sharp 1{·} indicators (and with the validity-vs-constraint Pareto behaviour our QM9 results show) is a clear under-explored slice. Remasking exists but has not been combined with an h-twist or with hard constraints. Backtracking / restart-with-buffer for CTMC samplers is essentially absent.

---

## Idea Bank

Ideas are scored as **N** (novelty vs above table, 1–5), **L** (expected lift in constraint sat., 1–5), **C** (implementation cost in days, ≤=cheap).

### 🏆 Idea 1 — Twisted-SMC for Hard Constraints with ESS-Adaptive Resampling (TS-FHS)

**N=4, L=5, C=1d.**

- **Hypothesis.** The h-twist gives the *optimal one-step proposal*; what's missing is the *correction* step of SMC. Maintain K particles per sample; at every CTMC step compute the importance weight w<sub>k</sub> ∝ Σ<sub>a</sub> p<sub>θ</sub>(a|x<sub>t</sub>)·h<sub>s</sub>(x<sup>−j</sup>⊕a) / h<sub>t</sub>(x<sub>t</sub>); resample (multinomial, stratified, or systematic) whenever ESS = (Σw)²/Σw² drops below 0.5·K.
- **Why it should win.** For binary constraints the unbiased SMC estimator of P(X<sub>0</sub>∈C) is exactly Π<sub>t</sub> w̄<sub>t</sub>; resampling concentrates compute on surviving particles instead of letting them silently die through the ε-floor. K=8 already buys ≈ 3× the chance to hit C per FHS step (back-of-envelope: 1−(1−p)<sup>K</sup> for independent successes).
- **What's new vs CDM/SMC-DDM.** Both target soft reward exp(βr(x)); we target the binary 1{X<sub>0</sub>∈C}. Hard constraints expose a different failure mode (entire-trajectory collapse rather than a softly skewed marginal) and need ESS-driven *aggressive* resampling and a buffer for dead particles. Also, neither competitor was evaluated on a Pareto frontier of *validity vs constraint sat*, which is the actual decision in this repo.
- **Validate by.** Run K∈{1,2,4,8,16} × ε∈{0,0.05,0.10,0.20} on QM9 with `constrained_fhs.py` extended to K particles. Expect monotone gain saturating around K=8.
- **Risk.** K× compute. Mitigation: share the B·V backbone forward over particles (they share most prefixes early).

### 🥈 Idea 2 — h-Aware ReMask Sampler (h-ReMDM)

**N=4, L=4, C=2d.**

- **Hypothesis.** FHS commits each position permanently; when a downstream step finds itself with all candidates < ε, the early commitments are to blame. *Re-mask* a position whose local marginal h<sub>φ</sub>(x with that position re-masked, t) is higher than current — i.e. take a step *backward* in the CTMC under a h-aware criterion. Repeat until clean.
- **Mechanism.** Augment the absorbing-state CTMC with re-mask rates μ<sup>−</sup><sub>j</sub>(x) chosen so the **detailed-balance-like** condition is preserved at the twisted level. Implementation: at each FHS step, with probability proportional to h-aware gain, re-mask a previously committed position and re-sample it under the current x.
- **Why it should win.** ReMDM ([2503.00307](https://arxiv.org/abs/2503.00307)) shows remasking lifts Pareto frontier even without h-twist. Combining remasking with the twist tackles the local-greedy failure mode that ε-thresholding cannot.
- **What's new.** ReMDM is unconditional / soft-guided; we provide the principled twisted-CTMC version with a chosen re-mask rate driven by h-φ gain and a CTMC-rigorous detailed-balance argument.
- **Validate by.** Compare against fixed-budget remask (random) and against vanilla FHS; expect higher validity *and* higher constraint sat (a Pareto move, not a trade).
- **Risk.** Convergence of the augmented CTMC needs a small proof (or empirical confirmation).

### 🥉 Idea 3 — Exact-h via Constraint-Structure Decomposition (TrieTwist)

**N=5, L=4, C=2–3d.**

- **Hypothesis.** The "novel-of-QM9" constraint is a *combinatorial* set: x ∉ S where S is a known finite set of canonical SMILES (133,885 of them). For partially-masked x, the **exact** h<sub>t</sub>(x) = P<sub>p<sub>θ</sub></sub>(X<sub>0</sub>∈C | X<sub>t</sub>=x) decomposes into a sum over completions, which can be computed exactly with a trie over S and the diffusion's clean-x marginals.
- **Mechanism.** Maintain a SMILES trie (or Bloom hash of canonical strings) over QM9-train. At each step compute h<sup>exact</sup><sub>t</sub>(x) = 1 − Σ<sub>s∈S, s compatible with x</sub> p<sub>θ</sub>(X<sub>0</sub>=s | X<sub>t</sub>=x). This is the *exact* h; the neural h<sub>φ</sub> can be either replaced or kept as a residual (h<sup>exact</sup> + neural residual for finer property control).
- **Why it should win.** Replaces the noisy, OOD-shifted neural h<sub>φ</sub> with a bias-free oracle for the novelty component. Removes the ε-threshold heuristic entirely. Calibration becomes trivial (h is a real probability).
- **What's new.** Nobody (in the surveyed literature) computes h exactly by exploiting the **structure of the forbidden set**. SVDD / SMC-DDM / CDM all assume a black-box reward. This is structurally cheaper and more accurate when the constraint admits a finite description.
- **Limitation.** Only works when the constraint is a finite forbidden set (novelty) or admits a tractable closed form. For QED-style continuous constraints, the neural h<sub>φ</sub> stays.
- **Validate by.** Drop in a trie of QM9-train, evaluate `constrained-fhs --use-exact-h`. Expect novel-of-valid → 95%+, validity unchanged (no false-pruning).
- **Risk.** p<sub>θ</sub>(s | x) for partial s requires a fast scoring routine — feasible because length=32 and 133k strings.

### Idea 4 — Constrained Gillespie / Exponential-Clock CTMC Sampler

**N=4, L=3, C=2d.**

- **Hypothesis.** All current samplers discretise the reverse CTMC with a fixed grid (T=32 steps). The *exact* CTMC simulation is via exponential clocks: at state x, draw τ ∼ Exp(Σ<sub>j,a</sub> r<sub>j</sub>(a|x)), and choose (j,a) with probability ∝ r<sub>j</sub>(a|x). With twist: r<sup>λ</sup><sub>j</sub>(a|x) = r<sub>j</sub>(a|x) · h<sub>s</sub>(x<sup>−j</sup>⊕a) / h<sub>t</sub>(x). Hard constraint: zero rates with h<sub>s</sub> < ε **before** drawing the clock, so dead candidates never even get a slot.
- **Why it should win.** Variance reduction: events with very small twisted rate (≈ε floor) almost never fire, but in the discrete-time grid they still consume a step. Gillespie gives adaptive step size for free. Subsumes Euler/DDPM/FHS as approximations.
- **What's new.** Stratified Hazard Sampling ([2601.02799](https://arxiv.org/html/2601.02799)) does min-variance event scheduling but **unconditional**; combining it with the h-twist *and* hard-pruning is open.
- **Validate by.** Compare `constrained-gillespie` against `constrained-fhs` at matched compute (NFE budget). Expect higher constraint sat at the same NFE.
- **Risk.** Variable wall-clock per sample (some take many more events).

### Idea 5 — Self-Calibrated h<sub>φ</sub> via Iterative On-Policy Retraining (DAg-Twist)

**N=3, L=3, C=1d/iteration × 3–5 iters.**

- **Hypothesis.** h<sub>φ</sub> is trained on **unconditional** FHS trajectories but evaluated on **twisted** trajectories. The distribution shift biases the twist. Iteratively (a) sample twisted trajectories with current h<sub>φ</sub>; (b) relabel each x<sub>t</sub> with its observed 1{X<sub>0</sub>∈C}; (c) retrain h<sub>φ</sub> with martingale-MSE on the new data. Fixed-point ≈ on-policy h.
- **What's new vs CDM.** CDM uses contrastive (pos/neg) training against the *amortised* twist; this is direct DAgger-style on-policy regression, requires no negative sample mining, and works for hard 1-indicators.
- **Validate by.** 3-iter loop on QM9; track h<sub>φ</sub>(x<sub>t</sub>,t) vs empirical P(X<sub>0</sub>∈C | x<sub>t</sub>) calibration error.
- **Risk.** Diminishing returns if h<sub>φ</sub> capacity is the bottleneck; iteration cost.

### Idea 6 — Multi-Step Bellman Twist (k-Lookahead h)

**N=3, L=2, C=1–2d.**

- **Hypothesis.** Current twist uses h<sub>s</sub>(x<sup>−j</sup>⊕a) — a *one-step* value. The proper Bellman backup is h<sup>(k)</sup><sub>s</sub>(x') = E[h<sub>s−kΔ</sub>(X<sub>s−kΔ</sub>) | X<sub>s</sub>=x'] for k≥1, sampled via k forward FHS rollouts from x'. Larger k ≈ smaller variance, better-aligned twist.
- **What's new.** SVDD does soft-value lookahead with a reward; the explicit hard-constraint Bellman version on top of CTMC FHS is not in the literature.
- **Validate by.** k∈{1,2,4} sweep on QM9; check constraint sat at fixed NFE.
- **Risk.** k× more h evaluations; only worth it if h<sub>φ</sub> is currently variance-bound.

### Idea 7 — Joint-Position Twist for Coupled Constraints (PairFHS)

**N=4, L=2, C=2d.**

- **Hypothesis.** FHS picks one masked position per step; but SMILES has strong cross-position dependencies (ring closures, valence balance). Committing one position at a time forces myopia. Generalise FHS to commit a *pair* of positions (or a small subset) jointly, with joint h<sub>s</sub>(x<sup>−{i,j}</sup>⊕(a,b)). The factored rate becomes coupled; theory: joint Doob over pairs.
- **Validate by.** PairFHS at NFE-matched cost (each pair-step ≈ 2 single-step costs).
- **Risk.** B·V² forwards per step → memory blow-up. Mitigation: restrict pairs to local neighbours.

### Idea 8 — Tempered Twist Schedule γ(t) (Soft-Hard CFG Analog)

**N=2, L=2, C=0.5d.**

- **Hypothesis.** Twist by h<sup>γ(t)</sup><sub>s</sub> with γ small at high noise (h is noisy → don't over-trust) and large at low noise (h is sharp). Analogous to the CFG γ schedule in continuous diffusion.
- **Validate by.** γ ∈ {const 1, linear, cosine} × ε sweep.
- **Risk.** Mostly engineering, not a research contribution by itself — but a free Pareto-frontier mover that should be reported anyway.

---

## Ranked Top Picks

| Rank | Idea                                | N | L | C    | Why                                                                |
| ---- | ----------------------------------- | - | - | ---- | ------------------------------------------------------------------ |
| 🏆 1  | **TS-FHS (twisted SMC + ESS)**      | 4 | 5 | 1d   | Largest expected lift; the cleanest theoretical gap.               |
| 🥈 2  | **h-Aware ReMask (h-ReMDM)**        | 4 | 4 | 2d   | Removes the local-greedy ε-failure mode; **orthogonal to TS-FHS**. |
| 🥉 3  | **TrieTwist (exact-h)**             | 5 | 4 | 2–3d | Highest novelty; eliminates OOD shift for the novelty constraint.  |
| 4    | Constrained Gillespie               | 4 | 3 | 2d   | Cleanest CTMC formulation; subsumes existing samplers.             |
| 5    | DAg-Twist (on-policy h retraining)  | 3 | 3 | 3–5d | Addresses calibration; slower payback.                             |

**Combine plan:** **TS-FHS + TrieTwist** gives both robustness (SMC) and bias-free h (trie). Then layer h-ReMask on top if there's still headroom. Expected QM9 constraint sat target: **70–80%** at K=8 particles + exact h. This is consistent with CoCoGraph (graph diff) hitting 98.5% novelty.

---

## Compute Budget Sketch (for the recommended pair)

| Variant                       | NFE / sample | GPU-hours @ 1000 samples | Note                                              |
| ----------------------------- | -----------: | ------------------------ | ------------------------------------------------- |
| `constrained-fhs` (current)   |        32·V  | ≈ 0.3                    | baseline                                          |
| TS-FHS, K=4                   |        32·V  | ≈ 1.2                    | particles share backbone forwards aggressively    |
| TS-FHS, K=8                   |        32·V  | ≈ 2.4                    | likely Pareto-best                                |
| TrieTwist (replacing h<sub>φ</sub>) |          32·V (no h-φ) | ≈ 0.2 | trie lookup ≪ network; **faster** than baseline   |
| TS-FHS + TrieTwist + h-ReMask |        ≈40·V | ≈ 3.0                    | full proposal                                     |

Within the **8 GPU-hour budget** for pilots.

---

## Open Risks & Eliminated Directions

**Eliminated early:**
- *Plain "particle guidance"-style repulsive guidance.* Designed for diversity in continuous diffusion (Corso et al.); the novelty constraint already covers the diversity axis. No gain.
- *Reward-based RL fine-tuning of the sampler.* Cost-prohibitive at this stage; we should exhaust training-free options first.
- *Pure inference-time-scaling (best-of-N).* Already implicitly being beaten by K-particle SMC (Idea 1).

**Risks of the recommended path:**
- TrieTwist needs a fast `p<sub>θ</sub>(s | x)` scorer for s ∈ QM9-train. For length-32 SMILES this is ≈ 133k forward evaluations per call; mitigated by batched scoring and trie pruning (only score s's whose prefix matches the committed positions).
- TS-FHS at K=8 increases memory; backbone activation sharing across particles is essential.
- h-ReMask CTMC needs a small mathematical write-up (detailed balance under the twisted target).

---

## Recommended Next Steps

1. **Pick 1–2 ideas** (recommended: TS-FHS + TrieTwist). Confirm direction.
2. Run `/novelty-check` on the chosen ideas — formal multi-source verification of novelty (current report is one-shot triage).
3. Run `/research-review` for external critique on the chosen design.
4. Run `/research-refine-pipeline` to produce a formal `FINAL_PROPOSAL.md` + `EXPERIMENT_PLAN.md`.
5. Then `/run-experiment` and `/auto-review-loop`.

---

## Files & References (selected)

- Existing code anchors: [constrained_ddpm.py](../constrained_ddpm.py), [constrained_euler.py](../constrained_euler.py), [constrained_fhs.py](../constrained_fhs.py), [discriminator_train.py](../discriminator_train.py).
- Baseline results: [sampling_eval.md](../sampling_eval.md).
- Anchor paper of repo: Schiff et al. 2024, [Simple Guidance Mechanisms for Discrete Diffusion Models](https://arxiv.org/abs/2412.10193).
- Closest concurrent work: [Test-Time Alignment of Discrete Diffusion via SMC (Zhao et al. ICLR 2025)](https://arxiv.org/abs/2505.22524), [CDM (cdm-smc.github.io)](https://cdm-smc.github.io/), [SVDD (Li et al. NeurIPS 2024)](https://arxiv.org/abs/2408.08252), [ReMDM (2503.00307)](https://arxiv.org/abs/2503.00307), [Constrained Discrete Diffusion (2503.09790)](https://arxiv.org/abs/2503.09790), [Stratified Hazard Sampling (2601.02799)](https://arxiv.org/html/2601.02799), [TPPF (2409.02399)](https://arxiv.org/abs/2409.02399).
