#!/bin/bash
# Baseline novelty points (run BEFORE deciding the adaptive-dual grid):
#   (1) no-guidance unconditional MDLM
#   (2) fixed-gamma D-CBG novelty guidance, gamma in {1,3,5}
# Headline metric = valid & novel count (compare vs CDD No-Guidance row:
#   MDLM 271 / UDLM 345 / CDD 511).
#
# Usage:
#   bash sample_novelty_baselines.sh [N_BATCHES] [BATCH_SIZE] [STEPS]
# Defaults: N_BATCHES=125 BATCH_SIZE=4 STEPS=128  (=> 500 samples, paper scale)

# NOTE: no `set -e` — one failing config must not abort the remaining baselines.
set -uo pipefail
cd /local/scratch/zhiheng/guidance

NUM_BATCHES=${1:-125}
BATCH_SIZE=${2:-4}
SAMPLING_STEPS=${3:-128}

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
mkdir -p "$OUT_DIR" /local/scratch/zhiheng/guidance/logs

STEPS_SUFFIX=""
if [ "$SAMPLING_STEPS" != "128" ]; then STEPS_SUFFIX="_steps${SAMPLING_STEPS}"; fi

run_eval () {
  # $1 = run_name ; rest = extra hydra overrides
  local RUN_NAME="$1"; shift
  local SAMPLES_JSON=$OUT_DIR/${RUN_NAME}_samples.json
  local CSV=$OUT_DIR/${RUN_NAME}_results.csv
  echo "[$(date)] === $RUN_NAME (N=$((NUM_BATCHES*BATCH_SIZE)) steps=$SAMPLING_STEPS GPU=$CUDA_VISIBLE_DEVICES) ==="
  $PY -u $EVAL \
    data=qm9 model=small backbone=dit model.length=32 \
    parameterization=subs diffusion=absorbing_state time_conditioning=False T=0 \
    training.guidance=null \
    eval.checkpoint_path=$DIFF_CKPT \
    sampling.steps=$SAMPLING_STEPS \
    sampling.num_sample_batches=$NUM_BATCHES \
    loader.eval_global_batch_size=$BATCH_SIZE \
    eval.generated_samples_path=$SAMPLES_JSON \
    +eval.results_csv_path=$CSV \
    seed=1 \
    hydra.run.dir=$OUT_DIR/novelty_eval_${RUN_NAME} \
    "$@"
  echo "[$(date)] done -> $CSV"
}

# (1) no-guidance unconditional MDLM (config default is `/guidance: null`, so
# pass NO guidance override — `guidance=null` is an invalid hydra group override).
run_eval "noguidance${STEPS_SUFFIX}"

# (2) fixed-gamma D-CBG novelty guidance
if [ ! -f "$CLF_CKPT" ]; then
  echo "[baselines] novelty classifier not found ($CLF_CKPT) — skipping gamma baselines." >&2
  echo "[baselines] train it first: bash scripts/train_novelty_classifier.sh"
  exit 0
fi
for G in 1 3 5; do
  run_eval "dcbg_gamma${G}${STEPS_SUFFIX}" \
    data.label_col=novel data.label_col_pctile=null data.num_classes=2 \
    classifier_backbone=dit classifier_model=tiny-classifier \
    guidance=cbg guidance.method=cbg guidance.gamma=$G guidance.condition=1 \
    guidance.classifier_checkpoint_path=$CLF_CKPT
done

echo "[$(date)] all baselines done. results in $OUT_DIR/*_results.csv"
