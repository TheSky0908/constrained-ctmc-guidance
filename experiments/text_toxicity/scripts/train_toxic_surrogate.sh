#!/bin/bash
# Fine-tune GPT-Neo on Jigsaw as the external toxicity violation scorer
# (CDD-style black-box evaluator used by tox_eval.py). Independent of the noisy
# DiT guidance classifier.
#
# Usage:
#   bash train_toxic_surrogate.sh [MODEL] [EPOCHS] [BATCH] [MAX_STEPS]
# Defaults: EleutherAI/gpt-neo-125M, 2 epochs, batch 16, full (max_steps=-1).
# Smoke:  bash train_toxic_surrogate.sh EleutherAI/gpt-neo-125M 1 8 50
#
# Output: outputs/owt/toxic_surrogate/  (HF model dir consumed by toxicity_scorer.py)

set -eo pipefail
cd /local/scratch/zhiheng/guidance

MODEL=${1:-EleutherAI/gpt-neo-1.3B}
EPOCHS=${2:-3}
BATCH=${3:-16}
MAX_STEPS=${4:--1}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/tan.1422/.cache/huggingface
export PYTHONPATH=/local/scratch/zhiheng/guidance

PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
OUT_DIR=/local/scratch/zhiheng/guidance/outputs/owt/toxic_surrogate
LOGDIR=/local/scratch/zhiheng/guidance/experiments/text_toxicity/logs
mkdir -p "$OUT_DIR" "$LOGDIR"
LOG=$LOGDIR/toxic_surrogate.log

exec > >(tee -a "$LOG") 2>&1   # self-log to experiments/text_toxicity/logs/
echo "[$(date)] fine-tuning $MODEL on Jigsaw (epochs=$EPOCHS, batch=$BATCH, max_steps=$MAX_STEPS) on GPU=$CUDA_VISIBLE_DEVICES"

$PY -u train_toxic_surrogate.py \
  --model "$MODEL" \
  --output_dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --max_steps "$MAX_STEPS"

echo "[$(date)] surrogate training done -> $OUT_DIR"
