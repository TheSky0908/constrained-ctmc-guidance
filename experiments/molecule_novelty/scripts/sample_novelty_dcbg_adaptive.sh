#!/bin/bash
# Sample MDLM with D-CBG using the adaptive-dual gamma schedule for the QM9
# NOVELTY task. The classifier predicts p(novel | x_t); the Lagrangian dual
# variable lambda is updated by projected gradient ascent each step:
#   lambda <- clip( (lambda - rho*(log p(novel|x_t) + C))_+ , 0, lambda_max )
#   x_{t+1} ~ p(.|x_t) * p(novel|.)^lambda
#
# Usage:
#   bash sample_novelty_dcbg_adaptive.sh [C] [RHO] [LAMBDA0] [LAMBDA_MAX] \
#                                         [N_BATCHES] [BATCH_SIZE] [TAG] [STEPS]
#
# Defaults: C=0.2231 RHO=0.5 LAMBDA0=0 LMAX=50 N_BATCHES=125 BS=4 adual 128

set -e
cd /local/scratch/zhiheng/guidance

C=${1:-0.2231}
RHO=${2:-0.5}
LAMBDA0=${3:-0.0}
LMAX=${4:-50.0}
NUM_BATCHES=${5:-125}
BATCH_SIZE=${6:-4}
TAG=${7:-adual}
SAMPLING_STEPS=${8:-128}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export HF_HOME=/local/scratch/zhiheng/guidance/.hf_cache
export PYTHONPATH=/local/scratch/zhiheng/guidance:/local/scratch/zhiheng/guidance/guidance_eval

PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
DIFF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt
CLF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/classifier/novelty_absorbing_state_T-0/checkpoints/best.ckpt
OUT_DIR=/local/scratch/zhiheng/guidance/experiments/molecule_novelty/results

STEPS_SUFFIX=""
if [ "$SAMPLING_STEPS" != "128" ]; then STEPS_SUFFIX="_steps${SAMPLING_STEPS}"; fi
RUN_NAME=mdlm_dcbg_novelty_${TAG}_C${C}_rho${RHO}_l0${LAMBDA0}_lmax${LMAX}${STEPS_SUFFIX}
SAMPLES_JSON=$OUT_DIR/${RUN_NAME}_samples.json
CSV=$OUT_DIR/${RUN_NAME}_results.csv

for p in "$DIFF_CKPT" "$CLF_CKPT"; do
  if [ ! -f "$p" ]; then
    echo "[sample_novelty_dcbg_adaptive] MISSING CHECKPOINT: $p" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR" /local/scratch/zhiheng/guidance/logs

echo "[$(date)] novelty adaptive_dual: C=$C rho=$RHO l0=$LAMBDA0 lmax=$LMAX GPU=$CUDA_VISIBLE_DEVICES"
echo "  N=$((NUM_BATCHES * BATCH_SIZE)) samples -> $RUN_NAME"

$PY -u experiments/molecule_novelty/novelty_eval.py \
  data=qm9 \
  model=small \
  backbone=dit \
  model.length=32 \
  parameterization=subs \
  diffusion=absorbing_state \
  time_conditioning=False \
  T=0 \
  training.guidance=null \
  data.label_col=novel \
  data.label_col_pctile=null \
  data.num_classes=2 \
  classifier_backbone=dit \
  classifier_model=tiny-classifier \
  eval.checkpoint_path=$DIFF_CKPT \
  guidance=cbg \
  guidance.method=cbg \
  guidance.gamma=1.0 \
  guidance.gamma_schedule=adaptive_dual \
  guidance.gamma_C=$C \
  guidance.gamma_rho=$RHO \
  guidance.gamma_lambda_init=$LAMBDA0 \
  guidance.gamma_lambda_max=$LMAX \
  guidance.condition=1 \
  guidance.classifier_checkpoint_path=$CLF_CKPT \
  sampling.steps=$SAMPLING_STEPS \
  sampling.num_sample_batches=$NUM_BATCHES \
  loader.eval_global_batch_size=$BATCH_SIZE \
  eval.generated_samples_path=$SAMPLES_JSON \
  +eval.results_csv_path=$CSV \
  seed=1 \
  hydra.run.dir=$OUT_DIR/novelty_eval_${RUN_NAME}

echo "[$(date)] novelty adaptive_dual sampling done. samples=$SAMPLES_JSON csv=$CSV"
