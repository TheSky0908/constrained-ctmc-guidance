#!/bin/bash
# Sample MDLM with D-CBG constant-γ guidance using the TIME-DEPENDENT SA
# classifier (time_conditioning=True), matching the classifier used by the
# adaptive-dual sample_first sweep. Base MDLM stays time-independent; only the
# classifier is time-conditioned (guidance.classifier_time_conditioning=True).
#
# Usage: bash sample_sa_dcbg_timedep.sh [GAMMA] [N_BATCHES] [BATCH_SIZE] [TAU] [STEPS] [SEED]
# Defaults: GAMMA=3, 125 batches x 4 = 500 samples, TAU=3.0, steps=128, seed=3

set -e
cd /local/scratch/zhiheng/guidance

GAMMA=${1:-3}
NUM_BATCHES=${2:-125}
BATCH_SIZE=${3:-4}
TAU=${4:-3.0}
SAMPLING_STEPS=${5:-128}
SEED=${6:-3}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export HF_HOME=/local/scratch/zhiheng/guidance/.hf_cache
export PYTHONPATH=/local/scratch/zhiheng/guidance:/local/scratch/zhiheng/guidance/guidance_eval

PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
DIFF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt
CLF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/classifier/sa_score_le_${TAU}_absorbing_state_T-0_timedep/checkpoints/best.ckpt
OUT_DIR=/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance

N=$((NUM_BATCHES * BATCH_SIZE))
STEPS_SUFFIX=""
if [ "$SAMPLING_STEPS" != "128" ]; then STEPS_SUFFIX="_steps${SAMPLING_STEPS}"; fi
SEED_SUFFIX=""
if [ "$SEED" != "1" ]; then SEED_SUFFIX="_seed${SEED}"; fi
RUN_NAME=mdlm_dcbg_sa_gamma${GAMMA}_td_n${N}_trainTau${TAU}${STEPS_SUFFIX}${SEED_SUFFIX}
SAMPLES_JSON=$OUT_DIR/${RUN_NAME}_samples.json
CSV=$OUT_DIR/${RUN_NAME}_results.csv

for p in "$DIFF_CKPT" "$CLF_CKPT"; do
  if [ ! -f "$p" ]; then
    echo "[sample_sa_dcbg_timedep] MISSING CHECKPOINT: $p" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR" /local/scratch/zhiheng/guidance/logs

echo "[$(date)] constant-γ + time-dep clf: γ=$GAMMA train-τ=$TAU steps=$SAMPLING_STEPS GPU=$CUDA_VISIBLE_DEVICES"
echo "  N=$N samples → $RUN_NAME"

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
  guidance.classifier_time_conditioning=True \
  guidance.condition=1 \
  guidance.classifier_checkpoint_path=$CLF_CKPT \
  sampling.steps=$SAMPLING_STEPS \
  sampling.num_sample_batches=$NUM_BATCHES \
  loader.eval_global_batch_size=$BATCH_SIZE \
  eval.generated_samples_path=$SAMPLES_JSON \
  +eval.results_csv_path=$CSV \
  seed=$SEED \
  hydra.run.dir=$OUT_DIR/sa_eval_${RUN_NAME}

echo "[$(date)] done γ=$GAMMA. samples=$SAMPLES_JSON csv=$CSV"
