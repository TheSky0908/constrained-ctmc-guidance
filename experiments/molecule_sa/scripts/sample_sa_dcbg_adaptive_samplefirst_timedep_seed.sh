#!/bin/bash
# Sample MDLM with adaptive-dual D-CBG using:
#   (1) the TIME-DEPENDENT SA classifier (time_conditioning=True), and
#   (2) the 'sample_first' dual-update order (paper Algorithm 1):
#         x_{t+1} ~ p(·|x_t) · p(y|·)^{λ_k}        # draw with current λ
#         λ_{k+1} = clip((λ_k - ρ·(log p(y|x_{t+1}) + C))_+, 0, λ_max)
#
# Contrast with sample_sa_dcbg_adaptive.sh, which uses the time-independent
# classifier and the default 'update_first' order.
#
# Usage:
#   bash sample_sa_dcbg_adaptive_samplefirst_timedep.sh [C] [RHO] [LAMBDA0] [LAMBDA_MAX] \
#                                    [N_BATCHES] [BATCH_SIZE] [TAU] [TAG] [STEPS]
#
# Defaults: C=0.2231 (=-log 0.8) RHO=0.5 LAMBDA0=0 LMAX=50 125 4 3.0 adual_sf_td 128

set -e
cd /local/scratch/zhiheng/guidance

C=${1:-0.2231}
RHO=${2:-0.5}
LAMBDA0=${3:-0.0}
LMAX=${4:-50.0}
NUM_BATCHES=${5:-125}
BATCH_SIZE=${6:-4}
TAU=${7:-3.0}
TAG=${8:-adual_sf_td}
SAMPLING_STEPS=${9:-128}
SEED=${10:-1}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export HF_HOME=/local/scratch/zhiheng/guidance/.hf_cache
export PYTHONPATH=/local/scratch/zhiheng/guidance:/local/scratch/zhiheng/guidance/guidance_eval

PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
DIFF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt
# Time-DEPENDENT classifier checkpoint (note the _timedep suffix).
CLF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/classifier/sa_score_le_${TAU}_absorbing_state_T-0_timedep/checkpoints/best.ckpt
OUT_DIR=/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance

STEPS_SUFFIX=""
if [ "$SAMPLING_STEPS" != "128" ]; then STEPS_SUFFIX="_steps${SAMPLING_STEPS}"; fi
SEED_SUFFIX=""
if [ "$SEED" != "1" ]; then SEED_SUFFIX="_seed${SEED}"; fi
RUN_NAME=mdlm_dcbg_sa_${TAG}_C${C}_rho${RHO}_l0${LAMBDA0}_lmax${LMAX}_trainTau${TAU}${STEPS_SUFFIX}${SEED_SUFFIX}
SAMPLES_JSON=$OUT_DIR/${RUN_NAME}_samples.json
CSV=$OUT_DIR/${RUN_NAME}_results.csv

for p in "$DIFF_CKPT" "$CLF_CKPT"; do
  if [ ! -f "$p" ]; then
    echo "[sample_sa_dcbg_adaptive_samplefirst_timedep] MISSING CHECKPOINT: $p" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR" /local/scratch/zhiheng/guidance/logs

echo "[$(date)] adual sample_first + time-dep clf: C=$C ρ=$RHO λ₀=$LAMBDA0 λmax=$LMAX train-τ=$TAU seed=$SEED GPU=$CUDA_VISIBLE_DEVICES"
echo "  N=$((NUM_BATCHES * BATCH_SIZE)) samples → $RUN_NAME"

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
  guidance.gamma=1.0 \
  guidance.gamma_schedule=adaptive_dual \
  guidance.gamma_adual_update_order=sample_first \
  guidance.classifier_time_conditioning=True \
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
  seed=$SEED \
  hydra.run.dir=$OUT_DIR/sa_eval_${RUN_NAME}

echo "[$(date)] adual sample_first + time-dep sampling done. samples=$SAMPLES_JSON csv=$CSV"
