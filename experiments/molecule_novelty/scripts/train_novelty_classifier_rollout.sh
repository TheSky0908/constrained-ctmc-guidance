#!/bin/bash
# Train the eq.(3) "pretrained-model path" novelty classifier r_phi(x_t, t) on
# the `qm9_novel_rollout` dataset: model-rollout intermediate states x_t labelled
# by the trajectory's terminal novelty. This is the TIME-DEPENDENT counterpart of
# train_novelty_classifier.sh — instead of forward-noising clean SMILES, the loss
# consumes (x_t, t) straight from the rollout dataset (training.classifier_rollout
# =True), and the classifier is fed the real noise level (time_conditioning=True).
#
# Prereq: build the rollout dataset first ->
#   bash experiments/molecule_novelty/scripts/build_rollout_dataset.sh
#
# Usage:
#   bash train_novelty_classifier_rollout.sh > logs/novelty_classifier_rollout.log 2>&1 &
#
# Output:
#   outputs/qm9/classifier/novelty_rollout_timedep/checkpoints/best.ckpt

set -e
cd /local/scratch/zhiheng/guidance

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export HF_HOME=/local/scratch/zhiheng/guidance/.hf_cache
export PYTHONPATH=/local/scratch/zhiheng/guidance:/local/scratch/zhiheng/guidance/guidance_eval

PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
RUN_NAME="novelty_rollout_timedep"
RUN_DIR=/local/scratch/zhiheng/guidance/outputs/qm9/classifier/$RUN_NAME
mkdir -p "$RUN_DIR" /local/scratch/zhiheng/guidance/logs

echo "[$(date)] starting ROLLOUT (eq.3) novelty classifier training on GPU=$CUDA_VISIBLE_DEVICES"
echo "[$(date)] run_dir=$RUN_DIR"

$PY -u -m main \
  mode=train_classifier \
  diffusion=absorbing_state \
  T=0 \
  parameterization=subs \
  time_conditioning=True \
  training.classifier_rollout=True \
  data=qm9_novel_rollout \
  data.label_col=novel \
  data.label_col_pctile=null \
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
  wandb.name=qm9-novelty-classifier-rollout \
  hydra.run.dir="$RUN_DIR"

echo "[$(date)] ROLLOUT novelty classifier training done"
