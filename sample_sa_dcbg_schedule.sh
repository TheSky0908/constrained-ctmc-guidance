#!/bin/bash
# Sample MDLM with D-CBG (classifier-based guidance) using a time-varying
# gamma(t) schedule. Companion to sample_sa_dcbg.sh.
#
# t convention: t=1 (start of sampling, fully noisy) -> t=0 (end, clean).
#
# Usage:
#   bash sample_sa_dcbg_schedule.sh [SCHEDULE] [GAMMA_MIN] [GAMMA_MAX] \
#                                    [N_BATCHES] [BATCH_SIZE] [TAU] [TAG]
#
# Schedules: linear_increasing, quadratic_increasing, linear_decreasing,
#            cosine_increasing, constant (uses GAMMA_MAX as constant gamma).
#
# Defaults: linear_increasing 1.0 5.0 125 4 3.0 sched

set -e
cd /local/scratch/zhiheng/guidance

SCHEDULE=${1:-linear_increasing}
GAMMA_MIN=${2:-1.0}
GAMMA_MAX=${3:-5.0}
NUM_BATCHES=${4:-125}
BATCH_SIZE=${5:-4}
TAU=${6:-3.0}
TAG=${7:-sched}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export HF_HOME=/local/scratch/zhiheng/guidance/.hf_cache
export PYTHONPATH=/local/scratch/zhiheng/guidance:/local/scratch/zhiheng/guidance/guidance_eval

PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
DIFF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt
CLF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/classifier/sa_score_le_${TAU}_absorbing_state_T-0/checkpoints/best.ckpt
OUT_DIR=/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance

RUN_NAME=mdlm_dcbg_sa_${TAG}_${SCHEDULE}_gmin${GAMMA_MIN}_gmax${GAMMA_MAX}_trainTau${TAU}
SAMPLES_JSON=$OUT_DIR/${RUN_NAME}_samples.json
CSV=$OUT_DIR/${RUN_NAME}_results.csv

for p in "$DIFF_CKPT" "$CLF_CKPT"; do
  if [ ! -f "$p" ]; then
    echo "[sample_sa_dcbg_schedule] MISSING CHECKPOINT: $p" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR" /local/scratch/zhiheng/guidance/logs

echo "[$(date)] starting MDLM + D-CBG schedule=$SCHEDULE gmin=$GAMMA_MIN gmax=$GAMMA_MAX train-tau=$TAU GPU=$CUDA_VISIBLE_DEVICES"
echo "  diffusion=$DIFF_CKPT"
echo "  classifier=$CLF_CKPT"
echo "  N=$((NUM_BATCHES * BATCH_SIZE)) samples"
echo "  output=$RUN_NAME"

$PY -u sa_eval.py \
  data=qm9 \
  model=small \
  backbone=dit \
  model.length=32 \
  parameterization=subs \
  diffusion=absorbing_state \
  time_conditioning=False \
  T=0 \
  training.guidance=null \
  data.label_col=sa_score \
  data.label_col_pctile=null \
  +data.label_col_value_threshold=$TAU \
  data.num_classes=2 \
  classifier_backbone=dit \
  classifier_model=tiny-classifier \
  eval.checkpoint_path=$DIFF_CKPT \
  guidance=cbg \
  guidance.method=cbg \
  guidance.gamma=$GAMMA_MAX \
  guidance.gamma_schedule=$SCHEDULE \
  guidance.gamma_min=$GAMMA_MIN \
  guidance.gamma_max=$GAMMA_MAX \
  guidance.condition=1 \
  guidance.classifier_checkpoint_path=$CLF_CKPT \
  sampling.steps=128 \
  sampling.num_sample_batches=$NUM_BATCHES \
  loader.eval_global_batch_size=$BATCH_SIZE \
  eval.generated_samples_path=$SAMPLES_JSON \
  +eval.results_csv_path=$CSV \
  seed=1 \
  hydra.run.dir=$OUT_DIR/sa_eval_${RUN_NAME}

echo "[$(date)] sampling done. samples=$SAMPLES_JSON csv=$CSV"
