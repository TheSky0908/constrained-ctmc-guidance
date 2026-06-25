#!/bin/bash
# Adaptive-dual D-CBG novelty sweep. Idempotent: skips combos whose result CSV
# already exists. Same scale as the gamma sweep (N=500, batch 4, steps 128).
#
# Each combo is "C RHO LAMBDA0 LMAX". Edit the COMBOS array below.
# adual update per step:  λ ← clip( (λ − ρ·(log p(novel|x_t) + C))₊ , 0, λ_max )
#   target novelty prob = e^{−C}.  C=0.2231 → p=0.80.
#
# Usage:  bash sweep_adual.sh [STEPS]   (default 128; filenames get _steps<N> when N≠128)

set -uo pipefail   # NO set -e: one failing combo must not abort the rest
cd /local/scratch/zhiheng/guidance

# Historical steps=128 grid (already done, CSVs on disk):
#   C=0.2231 ρ0.5 λmax{2,3,5,10,20,50}; C=0.01005 ρ{0.5,0.2} λmax{2,3,5,10}
#
# Active grid — steps=1024: all 512 adual params
#   C{-log0.99=0.01005, -log0.8=0.2231, -log0.5=0.6931} × ρ{0.5,0.1} × λmax{5,10}
COMBOS=(
  "0.01005 0.5 0.0 5.0"
  "0.01005 0.5 0.0 10.0"
  "0.01005 0.1 0.0 5.0"
  "0.01005 0.1 0.0 10.0"
  "0.2231 0.5 0.0 5.0"
  "0.2231 0.5 0.0 10.0"
  "0.2231 0.1 0.0 5.0"
  "0.2231 0.1 0.0 10.0"
  "0.6931 0.5 0.0 5.0"
  "0.6931 0.5 0.0 10.0"
  "0.6931 0.1 0.0 5.0"
  "0.6931 0.1 0.0 10.0"
)

NUM_BATCHES=125
BATCH_SIZE=4
SAMPLING_STEPS=${1:-128}
SUF=""; [ "$SAMPLING_STEPS" != "128" ] && SUF="_steps${SAMPLING_STEPS}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export HF_HOME=/local/scratch/zhiheng/guidance/.hf_cache
export PYTHONPATH=/local/scratch/zhiheng/guidance:/local/scratch/zhiheng/guidance/guidance_eval

PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
DIFF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt
CLF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/classifier/novelty_absorbing_state_T-0/checkpoints/best.ckpt
OUT_DIR=/local/scratch/zhiheng/guidance/experiments/molecule_novelty/results
EVAL=experiments/molecule_novelty/novelty_eval.py
LOGDIR=/local/scratch/zhiheng/guidance/experiments/molecule_novelty/logs
mkdir -p "$OUT_DIR" "$LOGDIR"

for p in "$DIFF_CKPT" "$CLF_CKPT"; do
  [ -f "$p" ] || { echo "[sweep_adual] MISSING CKPT: $p" >&2; exit 1; }
done

LOG=$LOGDIR/sweep_adual${SUF}.log
echo "[$(date '+%F %T')] sweep_adual start: ${#COMBOS[@]} combos steps=$SAMPLING_STEPS GPU=$CUDA_VISIBLE_DEVICES"
for combo in "${COMBOS[@]}"; do
  read -r C RHO L0 LMAX <<< "$combo"
  RUN_NAME="mdlm_dcbg_novelty_adual_C${C}_rho${RHO}_l0${L0}_lmax${LMAX}${SUF}"
  CSV=$OUT_DIR/${RUN_NAME}_results.csv
  if [ -f "$CSV" ]; then
    echo "[$(date '+%F %T')] [skip] $RUN_NAME (exists)"
    continue
  fi
  echo "[$(date '+%F %T')] === adual C=$C ρ=$RHO λ0=$L0 λmax=$LMAX (N=$((NUM_BATCHES*BATCH_SIZE))) ==="
  $PY -u $EVAL \
    data=qm9 model=small backbone=dit model.length=32 \
    parameterization=subs diffusion=absorbing_state time_conditioning=False T=0 \
    training.guidance=null \
    data.label_col=novel data.label_col_pctile=null data.num_classes=2 \
    classifier_backbone=dit classifier_model=tiny-classifier \
    eval.checkpoint_path=$DIFF_CKPT \
    guidance=cbg guidance.method=cbg guidance.gamma=1.0 \
    guidance.gamma_schedule=adaptive_dual \
    guidance.gamma_C=$C guidance.gamma_rho=$RHO \
    guidance.gamma_lambda_init=$L0 guidance.gamma_lambda_max=$LMAX \
    guidance.condition=1 guidance.classifier_checkpoint_path=$CLF_CKPT \
    sampling.steps=$SAMPLING_STEPS \
    sampling.num_sample_batches=$NUM_BATCHES \
    loader.eval_global_batch_size=$BATCH_SIZE \
    eval.generated_samples_path=$OUT_DIR/${RUN_NAME}_samples.json \
    +eval.results_csv_path=$CSV \
    seed=1 \
    hydra.run.dir=$OUT_DIR/novelty_eval_${RUN_NAME} \
    >> "$LOG" 2>&1
  RC=$?
  if [ $RC -eq 0 ] && [ -f "$CSV" ]; then
    echo "[$(date '+%F %T')] [done] λmax=$LMAX -> $(tail -1 "$CSV")"
  else
    echo "[$(date '+%F %T')] [FAIL] λmax=$LMAX (rc=$RC) — see $LOG"
  fi
done
echo "[$(date '+%F %T')] sweep_adual done."
