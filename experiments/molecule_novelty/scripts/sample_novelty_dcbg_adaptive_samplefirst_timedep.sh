#!/bin/bash
# Sample MDLM with adaptive-dual D-CBG on the NOVELTY task using:
#   (1) the ROLLOUT / TIME-DEPENDENT novelty classifier (novelty_rollout_timedep,
#       fed real sigma_t via classifier_time_conditioning=True), and
#   (2) the 'sample_first' dual-update order (paper Algorithm 1):
#         x_{t+1} ~ p(.|x_t) * p(novel|.)^{lambda_k}     # draw with current lambda
#         lambda_{k+1} = clip((lambda_k - rho*(log p(novel|x_{t+1}) + C))_+, 0, lambda_max)
#
# Novelty analogue of molecule_sa/scripts/sample_sa_dcbg_adaptive_samplefirst_timedep.sh.
# Records its own end-to-end wall-clock (seconds) and prints it.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash sample_novelty_dcbg_adaptive_samplefirst_timedep.sh \
#       [C] [RHO] [LAMBDA0] [LAMBDA_MAX] [N_BATCHES] [BATCH_SIZE] [TAG] [STEPS]
# Defaults: C=0.10536(=-log0.9) RHO=0.1 LAMBDA0=0 LMAX=5 125 4 adual_sf_td 64

set -uo pipefail
cd /local/scratch/zhiheng/guidance

C=${1:-0.10536}
RHO=${2:-0.1}
LAMBDA0=${3:-0.0}
LMAX=${4:-5.0}
NUM_BATCHES=${5:-125}
BATCH_SIZE=${6:-4}
TAG=${7:-adual_sf_td}
SAMPLING_STEPS=${8:-64}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export HF_HOME=/local/scratch/zhiheng/guidance/.hf_cache
export PYTHONPATH=/local/scratch/zhiheng/guidance:/local/scratch/zhiheng/guidance/guidance_eval

PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
DIFF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt
CLF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/classifier/novelty_rollout_timedep/checkpoints/best.ckpt
OUT_DIR=/local/scratch/zhiheng/guidance/experiments/molecule_novelty/results
EVAL=experiments/molecule_novelty/novelty_eval.py
LOGDIR=/local/scratch/zhiheng/guidance/experiments/molecule_novelty/logs
mkdir -p "$OUT_DIR" "$LOGDIR"

SUF=""; [ "$SAMPLING_STEPS" != "128" ] && SUF="_steps${SAMPLING_STEPS}"
RUN_NAME=mdlm_dcbg_novelty_${TAG}_C${C}_rho${RHO}_l0${LAMBDA0}_lmax${LMAX}${SUF}
SAMPLES_JSON=$OUT_DIR/${RUN_NAME}_samples.json
CSV=$OUT_DIR/${RUN_NAME}_results.csv
LOG=$LOGDIR/${RUN_NAME}.log

for p in "$DIFF_CKPT" "$CLF_CKPT"; do
  [ -f "$p" ] || { echo "[sample_novelty_adual_sf_td] MISSING CKPT: $p" >&2; exit 1; }
done

echo "[$(date '+%F %T')] adual sf+td (rollout/td clf): C=$C rho=$RHO l0=$LAMBDA0 lmax=$LMAX steps=$SAMPLING_STEPS GPU=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"
echo "  N=$((NUM_BATCHES * BATCH_SIZE)) -> $RUN_NAME" | tee -a "$LOG"

T0=$(date +%s)
$PY -u $EVAL \
  data=qm9 model=small backbone=dit model.length=32 \
  parameterization=subs diffusion=absorbing_state time_conditioning=False T=0 \
  training.guidance=null \
  data.label_col=novel data.label_col_pctile=null data.num_classes=2 \
  classifier_backbone=dit classifier_model=tiny-classifier \
  eval.checkpoint_path=$DIFF_CKPT \
  guidance=cbg guidance.method=cbg guidance.gamma=1.0 \
  guidance.gamma_schedule=adaptive_dual \
  guidance.gamma_adual_update_order=sample_first \
  guidance.classifier_time_conditioning=True \
  guidance.gamma_C=$C guidance.gamma_rho=$RHO \
  guidance.gamma_lambda_init=$LAMBDA0 guidance.gamma_lambda_max=$LMAX \
  guidance.condition=1 guidance.classifier_checkpoint_path=$CLF_CKPT \
  sampling.steps=$SAMPLING_STEPS \
  sampling.num_sample_batches=$NUM_BATCHES \
  loader.eval_global_batch_size=$BATCH_SIZE \
  eval.generated_samples_path=$SAMPLES_JSON \
  +eval.results_csv_path=$CSV \
  seed=1 \
  hydra.run.dir=$OUT_DIR/novelty_eval_${RUN_NAME} \
  >> "$LOG" 2>&1
RC=$?
SEC=$(( $(date +%s) - T0 ))
if [ $RC -eq 0 ] && [ -f "$CSV" ]; then
  echo "[$(date '+%F %T')] [done] time=${SEC}s -> $(tail -1 "$CSV")" | tee -a "$LOG"
else
  echo "[$(date '+%F %T')] [FAIL] rc=$RC time=${SEC}s — see $LOG" | tee -a "$LOG"
fi
echo "TIME_SECONDS=$SEC" | tee -a "$LOG"
