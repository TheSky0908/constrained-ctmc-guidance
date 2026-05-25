#!/bin/bash
# Sample MDLM with D-CBG (classifier-based guidance), conditioning on the
# "SA <= TAU" class using a TAU-specific trained classifier. Reproduces (and
# extends) CDD paper Fig 4 LEFT row "MDLM D-CBG g=3".
#
# Usage:
#   bash sample_sa_dcbg.sh [GAMMA] [N_BATCHES] [BATCH_SIZE] [TAU]
# Defaults: GAMMA=3, 125 batches x 8 = 1000 samples, TAU=3.0
#
# Prereqs:
#   - Diffusion ckpt:  outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt
#   - SA classifier:   outputs/qm9/classifier/sa_score_le_<TAU>_absorbing_state_T-0/checkpoints/best.ckpt

set -e
cd /local/scratch/zhiheng/guidance

GAMMA=${1:-3}
NUM_BATCHES=${2:-125}
BATCH_SIZE=${3:-8}
TAU=${4:-3.0}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export HF_HOME=/local/scratch/zhiheng/guidance/.hf_cache
export PYTHONPATH=/local/scratch/zhiheng/guidance:/local/scratch/zhiheng/guidance/guidance_eval

PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
DIFF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt
CLF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/classifier/sa_score_le_${TAU}_absorbing_state_T-0/checkpoints/best.ckpt
OUT_DIR=/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance
SAMPLES_JSON=$OUT_DIR/mdlm_dcbg_sa_gamma${GAMMA}_trainTau${TAU}_samples.json
CSV=$OUT_DIR/mdlm_dcbg_sa_gamma${GAMMA}_trainTau${TAU}_results.csv

# Sanity-check paths
for p in "$DIFF_CKPT" "$CLF_CKPT"; do
  if [ ! -f "$p" ]; then
    echo "[sample_sa_dcbg] MISSING CHECKPOINT: $p" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR" /local/scratch/zhiheng/guidance/logs

echo "[$(date)] starting MDLM + D-CBG (gamma=$GAMMA, train-tau=$TAU) on GPU=$CUDA_VISIBLE_DEVICES"
echo "  diffusion=$DIFF_CKPT"
echo "  classifier=$CLF_CKPT"
echo "  N=$((NUM_BATCHES * BATCH_SIZE)) samples"

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
  guidance.gamma=$GAMMA \
  guidance.condition=1 \
  guidance.classifier_checkpoint_path=$CLF_CKPT \
  sampling.steps=128 \
  sampling.num_sample_batches=$NUM_BATCHES \
  loader.eval_global_batch_size=$BATCH_SIZE \
  eval.generated_samples_path=$SAMPLES_JSON \
  +eval.results_csv_path=$CSV \
  seed=1 \
  hydra.run.dir=$OUT_DIR/sa_eval_gamma${GAMMA}_trainTau${TAU}

echo "[$(date)] sampling done (tau=$TAU). samples=$SAMPLES_JSON csv=$CSV"
