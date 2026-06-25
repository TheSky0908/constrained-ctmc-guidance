#!/bin/bash
# Replicate the FULL novelty set at a given sampling.steps:
#   no-guidance + CBG γ=1..10 + adaptive-dual COMBOS.
# Idempotent (skips existing CSVs). Filenames carry a `_steps<N>` suffix when
# N≠128 so the steps=128 results are never overwritten.
#
# Usage:  bash sweep_all.sh [STEPS] [GPU]
#   STEPS default 256, GPU default 1.

set -uo pipefail
cd /local/scratch/zhiheng/guidance

STEPS=${1:-256}
GPU=${2:-1}
NUM_BATCHES=125
BATCH_SIZE=4

GAMMAS="1 2 3 4 5 6 7 8 9 10"
ADUAL_COMBOS=(
  # C RHO L0 LMAX   (C=-log(0.8)=0.2231 ; C=-log(0.99)=0.01005)
  "0.2231 0.5 0.0 2.0"
  "0.2231 0.5 0.0 3.0"
  "0.2231 0.5 0.0 5.0"
  "0.2231 0.5 0.0 10.0"
  "0.2231 0.5 0.0 20.0"
  "0.2231 0.5 0.0 50.0"
  "0.01005 0.5 0.0 2.0"
  "0.01005 0.5 0.0 3.0"
  "0.01005 0.5 0.0 5.0"
  "0.01005 0.5 0.0 10.0"
  "0.01005 0.2 0.0 2.0"
  "0.01005 0.2 0.0 3.0"
  "0.01005 0.2 0.0 5.0"
  "0.01005 0.2 0.0 10.0"
)

export CUDA_VISIBLE_DEVICES=$GPU
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export HF_HOME=/local/scratch/zhiheng/guidance/.hf_cache
export PYTHONPATH=/local/scratch/zhiheng/guidance:/local/scratch/zhiheng/guidance/guidance_eval

PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
DIFF=/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt
CLF=/local/scratch/zhiheng/guidance/outputs/qm9/classifier/novelty_absorbing_state_T-0/checkpoints/best.ckpt
OUT=/local/scratch/zhiheng/guidance/experiments/molecule_novelty/results
EVAL=experiments/molecule_novelty/novelty_eval.py
LOG=/local/scratch/zhiheng/guidance/experiments/molecule_novelty/logs/sweep_all_steps${STEPS}.log
mkdir -p "$OUT"
for p in "$DIFF" "$CLF"; do [ -f "$p" ] || { echo "MISSING CKPT $p" >&2; exit 1; }; done

SUF=""; [ "$STEPS" != "128" ] && SUF="_steps${STEPS}"

run () {  # $1 run_name ; rest = extra hydra overrides
  local NAME="$1"; shift
  local CSV=$OUT/${NAME}_results.csv
  if [ -f "$CSV" ]; then echo "[$(date '+%F %T')] [skip] $NAME"; return; fi
  echo "[$(date '+%F %T')] === $NAME (steps=$STEPS) ==="
  $PY -u $EVAL \
    data=qm9 model=small backbone=dit model.length=32 \
    parameterization=subs diffusion=absorbing_state time_conditioning=False T=0 \
    training.guidance=null eval.checkpoint_path=$DIFF \
    sampling.steps=$STEPS sampling.num_sample_batches=$NUM_BATCHES \
    loader.eval_global_batch_size=$BATCH_SIZE \
    eval.generated_samples_path=$OUT/${NAME}_samples.json \
    +eval.results_csv_path=$CSV \
    seed=1 hydra.run.dir=$OUT/novelty_eval_${NAME} \
    "$@" >> "$LOG" 2>&1
  if [ $? -eq 0 ] && [ -f "$CSV" ]; then echo "[$(date '+%F %T')] [done] $NAME -> $(tail -1 "$CSV")"
  else echo "[$(date '+%F %T')] [FAIL] $NAME (see $LOG)"; fi
}

CLF_OV="data.label_col=novel data.label_col_pctile=null data.num_classes=2 classifier_backbone=dit classifier_model=tiny-classifier"

echo "[$(date '+%F %T')] sweep_all start: steps=$STEPS GPU=$GPU"

# (1) no-guidance
run "noguidance${SUF}"

# (2) CBG γ=1..10
for G in $GAMMAS; do
  run "dcbg_gamma${G}${SUF}" $CLF_OV \
    guidance=cbg guidance.method=cbg guidance.gamma=$G guidance.condition=1 \
    guidance.classifier_checkpoint_path=$CLF
done

# (3) adaptive-dual combos
for combo in "${ADUAL_COMBOS[@]}"; do
  read -r C RHO L0 LMAX <<< "$combo"
  run "mdlm_dcbg_novelty_adual_C${C}_rho${RHO}_l0${L0}_lmax${LMAX}${SUF}" $CLF_OV \
    guidance=cbg guidance.method=cbg guidance.gamma=1.0 \
    guidance.gamma_schedule=adaptive_dual \
    guidance.gamma_C=$C guidance.gamma_rho=$RHO \
    guidance.gamma_lambda_init=$L0 guidance.gamma_lambda_max=$LMAX \
    guidance.condition=1 guidance.classifier_checkpoint_path=$CLF
done

echo "[$(date '+%F %T')] sweep_all done (steps=$STEPS)."
