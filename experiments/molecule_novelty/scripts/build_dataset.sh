#!/bin/bash
# Build the qm9_novel on-disk dataset for the novelty classifier:
# generate valid SMILES from the frozen base MDLM, label by QM9-train
# membership (novel=1 / rediscovered=0), mix in QM9-train as label-0.
#
# Usage:
#   bash build_dataset.sh [NUM_VALID] [GEN_BATCH_SIZE]
#     NUM_VALID       target # unique valid generated molecules (default 50000)
#     GEN_BATCH_SIZE  generation batch size (default 512)
#
# Output: .data_cache/qm9_novelty_scored  (HF DatasetDict {train, validation})

set -e
cd /local/scratch/zhiheng/guidance

NUM_VALID=${1:-50000}
GEN_BS=${2:-512}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export HF_HOME=/local/scratch/zhiheng/guidance/.hf_cache
export PYTHONPATH=/local/scratch/zhiheng/guidance:/local/scratch/zhiheng/guidance/guidance_eval

PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
DIFF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt
OUT_DIR=/local/scratch/zhiheng/guidance/.data_cache/qm9_novelty_scored

if [ ! -f "$DIFF_CKPT" ]; then
  echo "[build_dataset] MISSING base ckpt: $DIFF_CKPT" >&2; exit 1
fi
mkdir -p /local/scratch/zhiheng/guidance/logs

echo "[$(date)] building qm9_novel dataset: num_valid=$NUM_VALID gen_bs=$GEN_BS GPU=$CUDA_VISIBLE_DEVICES"

$PY -u experiments/molecule_novelty/build_novelty_dataset.py \
  data=qm9 \
  model=small \
  backbone=dit \
  model.length=32 \
  parameterization=subs \
  diffusion=absorbing_state \
  time_conditioning=False \
  T=0 \
  training.guidance=null \
  eval.checkpoint_path=$DIFF_CKPT \
  +build.num_valid=$NUM_VALID \
  +build.gen_batch_size=$GEN_BS \
  +build.out_dir=$OUT_DIR \
  seed=1

echo "[$(date)] build done -> $OUT_DIR"
