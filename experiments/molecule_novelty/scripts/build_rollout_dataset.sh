#!/bin/bash
# Build the pretrained-model-rollout dataset for the eq.(3) novelty classifier.
# Draws unconditional trajectories from the frozen base MDLM, snapshots (x_t, t)
# along each, and labels every snapshot by the trajectory's terminal novelty.
# Output feeds the `qm9_novel_rollout` dataloader branch.
#
# Usage:
#   bash build_rollout_dataset.sh > logs/build_rollout.log 2>&1 &
#
# Output:
#   .data_cache/qm9_novelty_rollout  ({train, validation})

set -e
cd /local/scratch/zhiheng/guidance

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export HF_HOME=/local/scratch/zhiheng/guidance/.hf_cache
export PYTHONPATH=/local/scratch/zhiheng/guidance:/local/scratch/zhiheng/guidance/guidance_eval

PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
DIFF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt
OUT_DIR=/local/scratch/zhiheng/guidance/.data_cache/qm9_novelty_rollout

NUM_TRAJ=${1:-8000}
SNAPS_PER_TRAJ=${2:-8}
STEPS=${3:-128}
GEN_BS=${4:-512}

mkdir -p /local/scratch/zhiheng/guidance/logs

echo "[$(date)] building rollout dataset on GPU=$CUDA_VISIBLE_DEVICES"
echo "  num_traj=$NUM_TRAJ snaps/traj=$SNAPS_PER_TRAJ steps=$STEPS -> $OUT_DIR"

$PY -u experiments/molecule_novelty/build_rollout_dataset.py \
  data=qm9 \
  model=small \
  backbone=dit \
  model.length=32 \
  parameterization=subs \
  diffusion=absorbing_state \
  time_conditioning=False \
  T=0 \
  training.guidance=null \
  sampling.steps=$STEPS \
  eval.checkpoint_path=$DIFF_CKPT \
  +build.num_traj=$NUM_TRAJ \
  +build.snapshots_per_traj=$SNAPS_PER_TRAJ \
  +build.gen_batch_size=$GEN_BS \
  +build.out_dir=$OUT_DIR \
  seed=1 \
  hydra.run.dir=$OUT_DIR/build_run

echo "[$(date)] rollout dataset build done -> $OUT_DIR"
