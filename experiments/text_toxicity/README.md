# Natural-language toxicity mitigation (RealToxicityPrompts)

Extends the constrained-guidance methods from the molecule-SA task to the CDD
paper's *Natural Language Toxicity Mitigation* setting: generate a continuation
of a (toxic) prompt prefix while steering it toward the **non-toxic** class.

Base model is the pretrained **`kuleshov-group/mdlm-owt`** (masked diffusion LM,
OpenWebText, GPT-2 tokenizer, V=50258), loaded through the `hf_mdlm` backbone
adapter (`models/hf_mdlm.py`). Because the vocabulary is ~50k, all guidance uses
the **first-order `use_approx=True`** D-CBG path (the exact `O(B·L·V)` expansion
is infeasible at this vocab — this is the NL-scaling that the CDD appendix notes
was never done for D-CBG).

## Methods (current scope)
- **D-CBG, constant γ** (baseline) — `sample_tox_cbg.sh`
- **adaptive-dual** (proposed) — `sample_tox_adaptive.sh`
- **CDD-ALM** (baseline) — *deferred* (a `cdd_alm` guidance branch will drop in).

Guidance uses the **per-τ classifiers** (one per threshold τ ∈ {0.25, 0.50,
0.75}, class 1 = surrogate-score ≤ τ = "acceptable"), so sampling guides toward
`condition=1`. Both D-CBG and adaptive-dual share the τ-matched classifier; set
`TAU=<0.25|0.50|0.75>` when launching. *(The old binary main classifier —
`toxic_absorbing_state_T-0`, class 1 = toxic, condition=0 — was dropped from the
pipeline on 2026-06-07; `train_toxicity_classifier.sh` is kept only as a
deprecated reference.)*

## Pipeline
1. **Per-τ guidance classifiers** (noisy, on mdlm-owt latent space, labelled by
   the GPT-Neo surrogate's continuous score):
   `scripts/run_tau_classifiers_pipeline.sh` (or `train_tau_classifier.sh <TAU>`)
   → `outputs/owt/classifier/toxicity_le_<TAU>_absorbing_state_T-0/checkpoints/best.ckpt`
2. **External violation scorer** (clean-text GPT-Neo, CDD-style black box):
   `scripts/train_toxic_surrogate.sh` → `outputs/owt/toxic_surrogate/` (used by `toxicity_scorer.py`)
3. **Generate + evaluate** (`tox_eval.py`, at repo root), `TAU` required:
   `TAU=0.50 scripts/sample_tox_cbg.sh [GAMMA]`,
   `TAU=0.50 scripts/sample_tox_adaptive.sh [C] [RHO] ...`

## Metrics
- **Violation rate** at τ ∈ {0.25, 0.50, 0.75}: fraction of continuations the
  GPT-Neo surrogate scores above τ.
- **PPL**: GPT-2-XL perplexity of prompt+continuation.
- **Coherence (LLM-judge)**: stubbed (returns NaN) — to be wired to a
  Gemini/Gemma judge later.

## Layout (self-contained, mirrors `experiments/molecule_sa/`)
- `scripts/` — per-τ classifier + surrogate training and the two sampling launchers
- `results/` — curated `*_results.csv` (**tracked**)
- `analysis/` — plotting / summary scripts
- `figures/` — generated plots
- `logs/` — every script self-logs here via `exec > >(tee -a ...)` (**git-ignored**)
- runtime model dirs / raw per-run outputs live in `outputs/owt/...` (**git-ignored**)

## Shared code (repo root, used by BOTH tasks — not duplicated)
The guidance algorithms themselves are **not** task-specific and live once at the
repo root; molecule-SA and text-toxicity both call them:
- `diffusion.py` — D-CBG (`_cbg_denoise`, `use_approx`), the `adaptive_dual`
  γ-schedule, and prefix-conditioned sampling (`sample(prefix_ids=)`).
- `main.py` / configs — classifier training (`mode=train_classifier`).

Text-task-specific code (also repo root): `tox_eval.py` (harness),
`toxicity_scorer.py` (external scorer), `train_toxic_surrogate.py` (GPT-Neo
finetune), `models/hf_mdlm.py` (base-model adapter), Jigsaw loader in
`dataloader.py`. The molecule counterpart harness is `sa_eval.py`.
