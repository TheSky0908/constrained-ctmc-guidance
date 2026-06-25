#!/bin/bash
# Self-contained orchestration (meant to run detached in tmux on the server, so
# it survives SSH disconnects / local-machine sleep):
#   1. wait for the GPT-Neo surrogate training to finish,
#   2. label held-out Jigsaw with the surrogate's continuous toxicity score,
#   3. train the three per-tau guidance classifiers (tau=0.25/0.50/0.75).
#
# All compute runs on the server. Progress is appended to
# experiments/text_toxicity/logs/tau_pipeline.log.
#
# Usage (typically launched via tmux):
#   tmux new-session -d -s tau_pipeline \
#     'bash experiments/text_toxicity/scripts/run_tau_classifiers_pipeline.sh'

cd /local/scratch/zhiheng/guidance
LOGDIR=experiments/text_toxicity/logs
mkdir -p "$LOGDIR"
PIPELOG=$LOGDIR/tau_pipeline.log
SURR_LOG=$LOGDIR/toxic_surrogate.log
SURR_DIR=outputs/owt/toxic_surrogate
SCORED_DIR=.data_cache/jigsaw_surrogate_scored
PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
export HF_HOME=/home/tan.1422/.cache/huggingface
export PYTHONPATH=/local/scratch/zhiheng/guidance
export TOKENIZERS_PARALLELISM=false

log() { echo "[$(date '+%F %T')] $*" | tee -a "$PIPELOG"; }

pick_gpu() {
  # Freest GPU among indices 0-3 (by free memory). Respects the GPU policy.
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2)} $1<4 {print $2, $1}' \
    | sort -rn | head -1 | awk '{print $2}'
}

log "=== tau-classifier pipeline started ==="

# ── 1. Wait for the surrogate to finish (success marker in its log) ──────────
log "waiting for surrogate to finish (looking for 'best val F1=' in $SURR_LOG)…"
WAITED=0
until grep -q "best val F1=" "$SURR_LOG" 2>/dev/null; do
  sleep 30
  WAITED=$((WAITED + 30))
  if [ "$WAITED" -ge 14400 ]; then
    log "ERROR: surrogate not done after 4h; aborting."; exit 1
  fi
done
log "surrogate finished. $(grep -a 'best val F1=' "$SURR_LOG" | tail -1)"
sleep 10  # let the model files flush to disk

if [ ! -f "$SURR_DIR/model.safetensors" ] \
   && [ ! -f "$SURR_DIR/model.safetensors.index.json" ] \
   && [ ! -f "$SURR_DIR/pytorch_model.bin" ] \
   && [ ! -f "$SURR_DIR/pytorch_model.bin.index.json" ]; then
  log "ERROR: surrogate weights missing at $SURR_DIR; aborting."; exit 1
fi

# ── 2. Pick a GPU and label held-out Jigsaw with the surrogate ──────────────
GPU=$(pick_gpu)
export CUDA_VISIBLE_DEVICES=$GPU
log "using GPU $GPU for labeling + classifier training"

if [ -d "$SCORED_DIR" ]; then
  log "scored dataset already exists at $SCORED_DIR; skipping labeling."
else
  log "labeling held-out Jigsaw test with surrogate (60k comments)…"
  $PY -u label_jigsaw_with_surrogate.py \
    --surrogate_dir "$SURR_DIR" --num 60000 --out_dir "$SCORED_DIR" \
    >> "$PIPELOG" 2>&1
  if [ ! -d "$SCORED_DIR" ]; then
    log "ERROR: labeling failed (no $SCORED_DIR); aborting."; exit 1
  fi
  log "labeling done."
fi

# ── 3. Train the three per-tau guidance classifiers ─────────────────────────
for TAU in 0.25 0.50 0.75; do
  log "training tau=$TAU classifier (max_steps=20000) on GPU $GPU…"
  CUDA_VISIBLE_DEVICES=$GPU bash experiments/text_toxicity/scripts/train_tau_classifier.sh \
    "$TAU" 20000 >> "$PIPELOG" 2>&1
  CKPT=outputs/owt/classifier/toxicity_le_${TAU}_absorbing_state_T-0/checkpoints/best.ckpt
  if [ -f "$CKPT" ]; then
    log "tau=$TAU done -> $CKPT"
  else
    log "WARNING: tau=$TAU produced no best.ckpt (check $PIPELOG)."
  fi
done

log "=== ALL DONE: 3 per-tau classifiers trained ==="
touch outputs/owt/classifier/.TAU_PIPELINE_DONE
