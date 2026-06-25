#!/bin/bash
# Train ONE per-tau noisy D-CBG / adaptive-dual guidance classifier on the
# surrogate-scored held-out Jigsaw text. Class 1 = (toxicity_score <= tau) =
# "acceptable at level tau" (guide toward condition=1). Mirrors the molecule-SA
# per-tau classifier (sa_score_le_<tau>), but the continuous score comes from
# the GPT-Neo surrogate (see label_jigsaw_with_surrogate.py).
#
# Usage:
#   bash train_tau_classifier.sh <TAU> [MAX_STEPS] [GLOBAL_BS]
# Defaults: MAX_STEPS=20000, GLOBAL_BS=256.
#
# Output:
#   outputs/owt/classifier/toxicity_le_<TAU>_absorbing_state_T-0/checkpoints/best.ckpt

set -e
cd /local/scratch/zhiheng/guidance

TAU=${1:?"give a toxicity threshold tau, e.g. 0.25"}
MAX_STEPS=${2:-20000}
GLOBAL_BS=${3:-256}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/tan.1422/.cache/huggingface
export PYTHONPATH=/local/scratch/zhiheng/guidance

PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
RUN_NAME="toxicity_le_${TAU}_absorbing_state_T-0"
RUN_DIR=/local/scratch/zhiheng/guidance/outputs/owt/classifier/$RUN_NAME
mkdir -p "$RUN_DIR" /local/scratch/zhiheng/guidance/experiments/text_toxicity/logs

echo "[$(date)] training per-tau toxicity classifier tau=$TAU on GPU=$CUDA_VISIBLE_DEVICES (max_steps=$MAX_STEPS)"
echo "[$(date)] run_dir=$RUN_DIR"

$PY -u -m main \
  mode=train_classifier \
  diffusion=absorbing_state \
  T=0 \
  parameterization=subs \
  time_conditioning=False \
  data=jigsaw_scored \
  data.label_col=toxicity_score \
  data.label_col_pctile=null \
  +data.label_col_value_threshold=$TAU \
  data.num_classes=2 \
  loader.global_batch_size=$GLOBAL_BS \
  loader.eval_global_batch_size=$GLOBAL_BS \
  loader.num_workers=0 \
  loader.persistent_workers=False \
  classifier_backbone=dit \
  classifier_model=tiny-classifier \
  model.length=128 \
  optim.lr=3e-4 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=1000 \
  lr_scheduler.lr_min=3e-6 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
  callbacks.checkpoint_monitor.monitor=val/cross_entropy \
  trainer.devices=1 \
  trainer.val_check_interval=1.0 \
  trainer.max_steps=$MAX_STEPS \
  wandb.group=train_classifier \
  wandb.name=jigsaw-toxicity-classifier-le${TAU} \
  hydra.run.dir="$RUN_DIR"

echo "[$(date)] tau=$TAU classifier done -> $RUN_DIR"
