# molecule_novelty — adaptive-dual D-CBG on QM9 novelty

Apply the **adaptive-dual D-CBG** sampler (same mechanism as `molecule_sa`) to
the QM9 **novelty** task from the CDD paper (NeurIPS 2025, arXiv:2503.09790,
Fig. 4 right). Headline KPI = number of **valid & novel** molecules (maximize).

CDD "No Guidance" reference (valid&novel / QED): AR 11/0.41 · MDLM 271/0.45 ·
UDLM 345/0.46 · CDD **511**/0.45.

## Key idea

The guidance classifier predicts **p(novel | x_t)** directly (class 1 = novel),
NOT QED. QM9-train is all label-0 ("not novel"), so we build the training set
offline (mirrors how `text_toxicity` built `jigsaw_scored`):

1. Generate valid SMILES from the frozen base MDLM (one-token-per-step FHS).
2. Canonicalize; label `novel=1` if absent from QM9-train, else `0`.
3. Mix in QM9-train molecules as guaranteed label-0 to balance classes (auto;
   at 50k scale the base model rediscovers enough QM9 that 0 extra mixing was
   needed — natural ~42% novel).

The adaptive-dual schedule then presses the novelty constraint via the dual
variable λ (projected gradient ascent on `log p(novel|x_t) + C` per step).

## Pipeline

```bash
# 1. Build the qm9_novel dataset  (-> .data_cache/qm9_novelty_scored)
bash scripts/build_dataset.sh 50000 512

# 2. Train the novelty classifier  (-> outputs/qm9/classifier/novelty_absorbing_state_T-0)
bash scripts/train_novelty_classifier.sh

# 3a. Baselines first: no-guidance + fixed gamma {1,3,5}
bash scripts/sample_novelty_baselines.sh

# 3b. Adaptive-dual sweep (decide grid from baselines)
bash scripts/sample_novelty_dcbg_adaptive.sh <C> <RHO> <LAMBDA0> <LMAX>

# 4. Eval is built into novelty_eval.py (Valid/Unique/Novel/QED + λ traj.json)
```

## Files

- `build_novelty_dataset.py` — generate + label + balance → on-disk HF dataset.
- `novelty_eval.py` — sample + evaluate (mirror of `../../sa_eval.py`); dumps
  adaptive_dual λ trajectory to `*_traj.json`.
- `scripts/` — build / train / baseline / adaptive-dual sampling drivers.
- `results/`, `figures/`, `logs/`, `analysis/` — artifacts.

## Repo touch-points (shared code)

- `dataloader.py`: added `qm9_novel` branch (load on-disk dataset; `'qm9' in
  name` reuses SMILES tokenization).
- `configs/data/qm9_novel.yaml`: data config (`label_col=novel`, num_classes=2).
