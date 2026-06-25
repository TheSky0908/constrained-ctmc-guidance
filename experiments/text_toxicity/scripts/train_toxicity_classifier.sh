#!/bin/bash
# ============================================================================
# DEPRECATED (2026-06-07): the binary main toxicity classifier has been dropped
# from the pipeline. Guidance now uses the per-tau classifiers only — train them
# with train_tau_classifier.sh / run_tau_classifiers_pipeline.sh. This script is
# kept for reference; its checkpoint (toxic_absorbing_state_T-0) is no longer
# used by the sampling scripts.
# ============================================================================
# Train the noisy D-CBG / adaptive-dual guidance classifier on Jigsaw with a
# binary toxicity label (class 1 = toxic, class 0 = non-toxic). The classifier
# operates on the SAME absorbing-state GPT-2 latent space as kuleshov-group/
# mdlm-owt (vocab 50258, mask 50257, T=0, time_conditioning=False), so its
# log p(y|x_t) can be used to guide mdlm-owt sampling.
#
# Usage:
#   bash train_toxicity_classifier.sh [MAX_STEPS] [GLOBAL_BS]
# Defaults: MAX_STEPS=25000, GLOBAL_BS=256. For a quick smoke: MAX_STEPS=50.
#
# Output:
#   outputs/owt/classifier/toxic_absorbing_state_T-0/checkpoints/best.ckpt

set -e
cd /local/scratch/zhiheng/guidance

MAX_STEPS=${1:-25000}
GLOBAL_BS=${2:-256}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
# Reuse the user HF cache (mdlm-owt + gpt2 tokenizer live here).
export HF_HOME=/home/tan.1422/.cache/huggingface
export PYTHONPATH=/local/scratch/zhiheng/guidance

PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
RUN_NAME="toxic_absorbing_state_T-0"
RUN_DIR=/local/scratch/zhiheng/guidance/outputs/owt/classifier/$RUN_NAME
mkdir -p "$RUN_DIR" /local/scratch/zhiheng/guidance/experiments/text_toxicity/logs

echo "[$(date)] training toxicity classifier on GPU=$CUDA_VISIBLE_DEVICES (max_steps=$MAX_STEPS, global_bs=$GLOBAL_BS)"
echo "[$(date)] run_dir=$RUN_DIR"

$PY -u -m main \
  mode=train_classifier \
  diffusion=absorbing_state \
  T=0 \
  parameterization=subs \
  time_conditioning=False \
  data=jigsaw \
  data.label_col=toxic \
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
  wandb.name=jigsaw-toxicity-classifier \
  hydra.run.dir="$RUN_DIR"

echo "[$(date)] toxicity classifier training done"
