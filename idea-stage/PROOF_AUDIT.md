# Proof Audit — A.5 "Online tracking under per-step drift"

Target: `adaptive_dual_guidance.tex`, subsection `\label{app:tracking}`
Reviewer: Codex `gpt-5.x`, reasoning effort `xhigh`, 3 rounds.
Scope: Lemmas `lem:trk-sens`, `lem:trk-contr`; Corollaries `cor:trk-path`,
`cor:trk-tradeoff`, `cor:trk-violation`; Theorems `thm:trk`, `thm:trk-regret`.
Supporting facts (`prop:smooth`, `prop:noise`, `as:floor`) assumed.

## Round 1 — 9 issues (verdict FAIL)
| id | severity | location | issue | fix |
|----|----------|----------|-------|-----|
| 1 | INVALID/GLOBAL | cor:trk-violation | boundary `|v_k|≤L|a_k|` false (counterexample λ*=0) | restated with positive part `(v_k)_+ ≤ L|a_k| + r_k`, `r_k=(g'(λ_max))_+` infeasibility residual; 3-case KKT proof |
| 2 | INVALID/GLOBAL | thm:trk-regret | cited `P_K=O(1)` from strong-concavity cor while dropping strong concavity | general `P_K`; rate only under added `P_K=o(K)`; alternating-linear counterexample shown |
| 3 | UNJUSTIFIED/GLOBAL | Notation, eq:trk-var | variance bound used unbiased-only Prop while allowing bias | proved `Var(ĝ'|F)≤R²/4` directly via Popoviciu (bounded range), unconditional on law |
| 4 | UNJUSTIFIED/LOCAL | thm:trk-regret | wrong telescope object | corrected via ±(λ_{k+1}-λ*_{k+1})² insertion, `2λ_max P_K` bound |
| 5 | UNJUSTIFIED/LOCAL | thm:trk-regret | "absorbed into G_lip²" hand-waving | theorem made deterministic full-gradient |
| 6 | UNJUSTIFIED/LOCAL | lem:trk-sens | divided by |Δ| without Δ=0 | added trivial Δ=0 case |
| 7 | UNDERSTATED/LOCAL | cor:trk-tradeoff | δ^{2/3} MSE vs RMSE; ρ≤1/L | MSE∝δ^{2/3}, RMSE∝δ^{1/3}, ρ* clamp noted |
| 8 | COSMETIC | cor:trk-path | nonuniform grid | dt_k, max_k dt_k, uniform constants |
| 9 | COSMETIC | Notation | m_k notation collision | renamed → \bar g_k |

## Round 2 — 7/9 resolved; 5 new (verdict FAIL)
| id | severity | issue | fix |
|----|----------|-------|-----|
| NEW-01 | INVALID/GLOBAL | regret rate: P_K=o(K) ↛ o(1) with ρ∝1/√K | optimal ρ*=Θ(√((1+P_K)/K)); avg regret O(√((1+P_K)/K)), o(1) iff P_K=o(K) |
| NEW-02 | INVALID/GLOBAL | F_k included x_k ⇒ ζ_k≡0 (martingale collapse) | F_k redefined as pre-sample σ(x_0..x_{k-1}); x_k drawn given F_k |
| NEW-03 | INVALID/LOCAL | "b_k=0 exactly when law=q" (full-law unnecessary) | "iff equal conditional expected surprisal; in particular law=q" |
| NEW-04 | COSMETIC | o_K(1) ≠ (1-ρμ)^K | Cesàro average = O(1/(Kρμ)) |
| NEW-05 | LOCAL | set-valued argmax | fix measurable maximizer selection |

## Round 3 — 5/5 resolved; 2 minor (verdict WARN) → both fixed
| id | severity | issue | fix |
|----|----------|-------|-----|
| R3-01 | UNDERSTATED/LOCAL | G_lip=R+C needs C≥0 | G_lip=R+|C| (note budget C≥0 ⇒ =R+C) |
| R3-02 | COSMETIC | divide by ρ needs ρ>0 | `0<ρ≤1/L` / `ρ>0` stated |

## Round 4 — unbiasedness rewrite (user correction, option A)
User corrected the indexing: `q_{λ_k} ∝ p(x|x_{k-1}) h(x)^{λ_k}` is the transform
that PRODUCES `x_k`, so the held sample `x_k` is an exact draw from it ⇒ the
single-sample gradient is **conditionally unbiased** (`b_k=0`), not biased. The
earlier rem:bias "lagged distribution" justification was wrong.

Rewrote: Notation (b_k=0 by construction, eq:trk-unbiased), thm:trk (floor
σ_g/μ + drift, no β̄), cor:trk-tradeoff (MSE ρσ_g²/μ + δ̄²/(ρμ)², no β̄²/μ²),
cor:trk-violation (κσ_g floor), rem:bias (demoted to ε-floor/learned-h
implementation caveat), rem:trk-caveats(ii). β̄ now appears only as an optional
implementation-bias add-on.

Codex re-review (round 4): C1–C3 RESOLVED (core unbiasedness correct); WARN with
4 bookkeeping items, all fixed:
- I3: `r_k` random ⇒ use `(1/K)Σ E[r_k]` in eq:trk-viol.
- I1: unbiasedness holds for `k≥1`; `x_0` (masked prior) `b_0`-bias absorbed in transient.
- C6: conditioning on `F_k` fixes both `λ_k` and `x_{k-1}` (whole program) — made explicit.
- I2: as:trk-drift strengthened to pathwise/a.s. bound absorbing the conditioning jump `x_{k-1}→x_k` (+ conditional-drift fallback).
- I4: rem:bias vanishing claim softened (needs `log h` Lipschitz / `h` bounded away from 0; kept qualitative).

## Round 5 — standing-assumption cleanup (well-trained ĥ + no floor ⇒ no bias)
Added Assumption `as:exact`: (a) exact soft tempering (ε-floor excluded from Alg 1)
⇒ `b_k=0` for k≥1 (unbiased for the ĥ-dual); (b) well-trained ĥ (η=‖log ĥ−log h*‖_∞)
⇒ enforced constraint = true reachability up to η. Removed the β̄ floor terms from
thm:trk / cor:trk-tradeoff / cor:trk-violation; compressed rem:bias to a
relaxation note; retitled app:sgd.

Codex re-review (round 5): no-β̄ algebra consistent; WARN with 5 scope/wording
items, all fixed (implementing the reviewer's own suggested fixes):
- R1: "coincides" overclaimed → (a) drives unbiasedness; (b) gives true-reachability **up to η** (exact iff η=0).
- R2: `as:floor` reworded — bounded-range hypothesis (for Popoviciu) distinct from the active ε-mask, which `as:exact`(a) excludes ⇒ realised law is exactly q_λ.
- R3: rem:bias split — ε-floor = genuine gradient bias of the ĥ-dual; inexact ĥ = constraint misspecification (η), not a gradient bias (+ law-shift term only vs the true h*-dual).
- R4: per-floor units made explicit — first-moment `+β̄/μ`, MSE `+O(β̄²/μ²)`, violation `+κβ̄`; batching shrinks variance, not the deterministic bias.
- R5: thm:trk `k=0` transient made explicit (recursion for k≥1; one-off `|b_0|≤R` as a geometric transient).

## Round 6 — drop the ε-floor from the theory ("treat as nonexistent")
Per user: remove the ε-floor entirely from the analysis. Edits: `as:floor` now an
**intrinsic** bounded-discriminator bound (h∈[ε,1], bounded-logit sigmoid), R not
a floor; `as:exact`(a) = exact sampling from q_λ (no floor to exclude) ⇒ b_k=0
(k≥1); `rem:bias` reduced to **misspecification-only** (imperfect ĥ is not a
gradient bias; shifts enforced constraint by ≤ η); **β̄ removed everywhere**
(no floor ⇒ no floor-bias); dropped the sec:primal floor-as-support sentence;
rem:trk-caveats(ii) simplified.

Codex re-review (round 6): **VERDICT PASS** — W1–W5 all OK, no blocking issues.
"The no-floor idealization is mathematically coherent under the intrinsic
bounded-log-discriminator and exact-sampling assumptions." Two non-blocking
polish items fixed:
- lem:mono prose "strictly increases" → "raises (strictly unless ĥ a.s. constant, i.e. Cov>0)".
- rem:trk-caveats(i) "floored μ_k" → "lower-bounded (truncated) μ_k" (avoid ε-floor collision).
Caveat noted (non-blocking): finite η=‖log ĥ−log h*‖_∞ implicitly needs h* positive on the support.

## Final status
Acceptance gate (zero open FATAL/CRITICAL): **MET** across all rounds. Core
results: per-step gradient conditionally unbiased; tracking floor σ_g/μ + drift;
δ̄^{2/3} MSE; dynamic regret O(√((1+P_K)/K)); constraint-violation κσ_g + drift +
E[r_k]. Reviewer confirmed unbiasedness argument, contraction, three-case
violation proof.
LaTeX not compiled (no toolchain installed); environment/display-math balance
verified by hand. **Result: PASS (round 6, no-floor theory — Codex verdict PASS).**
