#!/bin/bash
# Train D-CBG classifier on QM9 with binary SA labels.
# class 1 = (SA <= TAU); class 0 = (SA > TAU).
#
# Usage:
#   bash train_sa_classifier.sh [TAU]
#     TAU defaults to 3.0
#
# Examples:
#   bash train_sa_classifier.sh 3.0 > logs/sa_classifier_le3.0.log 2>&1 &
#   bash train_sa_classifier.sh 4.5 > logs/sa_classifier_le4.5.log 2>&1 &
#
# Outputs:
#   outputs/qm9/classifier/sa_score_le_<TAU>_absorbing_state_T-0/checkpoints/best.ckpt

set -e
cd /local/scratch/zhiheng/guidance

TAU=${1:-3.0}

# --- GPU + env ---
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export HF_HOME=/local/scratch/zhiheng/guidance/.hf_cache
export PYTHONPATH=/local/scratch/zhiheng/guidance:/local/scratch/zhiheng/guidance/guidance_eval

PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
RUN_NAME="sa_score_le_${TAU}_absorbing_state_T-0"
RUN_DIR=/local/scratch/zhiheng/guidance/outputs/qm9/classifier/$RUN_NAME
mkdir -p "$RUN_DIR" /local/scratch/zhiheng/guidance/logs

echo "[$(date)] starting SA classifier training (tau=$TAU) on GPU=$CUDA_VISIBLE_DEVICES"
echo "[$(date)] run_dir=$RUN_DIR"

$PY -u -m main \
  mode=train_classifier \
  diffusion=absorbing_state \
  T=0 \
  parameterization=subs \
  time_conditioning=False \
  data=qm9 \
  data.label_col=sa_score \
  data.label_col_pctile=null \
  +data.label_col_value_threshold=$TAU \
  data.num_classes=2 \
  loader.global_batch_size=2048 \
  loader.eval_global_batch_size=4096 \
  loader.num_workers=0 \
  loader.persistent_workers=False \
  classifier_backbone=dit \
  classifier_model=tiny-classifier \
  model.length=32 \
  optim.lr=3e-4 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=1000 \
  lr_scheduler.lr_min=3e-6 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
  callbacks.checkpoint_monitor.monitor=val/cross_entropy \
  trainer.devices=1 \
  trainer.val_check_interval=1.0 \
  trainer.max_steps=25000 \
  wandb.group=train_classifier \
  wandb.name=qm9-sa-classifier_le${TAU} \
  hydra.run.dir="$RUN_DIR"

echo "[$(date)] training done (tau=$TAU)"
