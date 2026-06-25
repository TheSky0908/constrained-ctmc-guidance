#!/bin/bash
# Constant-gamma D-CBG novelty sweep using the eq.(3) ROLLOUT classifier
# (novelty_rollout_timedep, time-DEPENDENT). Mirrors sweep_gamma.sh exactly
# (N=500, batch 4, steps 128, seed 1) EXCEPT:
#   - classifier ckpt -> novelty_rollout_timedep/best.ckpt
#   - guidance.classifier_time_conditioning=True (feed real sigma_t to the
#     time-dependent discriminator; base MDLM stays time_conditioning=False)
#   - output filenames carry a `_rollout` tag so they never collide with / skip
#     the existing forward-classifier results.
# Idempotent: skips any gamma whose result CSV already exists.
#
# Usage:
#   bash sweep_gamma_rollout.sh ["0 1 2 ... 10"] [STEPS]
#   (set CUDA_VISIBLE_DEVICES to pick the GPU; split the gamma list across cards
#    to parallelize, e.g. GPU2 "0 1 2 3 4 5" + GPU3 "6 7 8 9 10".)

set -uo pipefail   # NO set -e: one failing gamma must not abort the rest
cd /local/scratch/zhiheng/guidance

GAMMAS=${1:-"0 1 2 3 4 5 6 7 8 9 10"}
SAMPLING_STEPS=${2:-128}
NUM_BATCHES=125
BATCH_SIZE=4
SUF=""; [ "$SAMPLING_STEPS" != "128" ] && SUF="_steps${SAMPLING_STEPS}"
TAG="rollout"

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

for p in "$DIFF_CKPT" "$CLF_CKPT"; do
  [ -f "$p" ] || { echo "[sweep_gamma_rollout] MISSING CKPT: $p" >&2; exit 1; }
done

LOG=$LOGDIR/sweep_gamma_${TAG}${SUF}_gpu${CUDA_VISIBLE_DEVICES}.log
echo "[$(date '+%F %T')] sweep_gamma_rollout start: gammas=[$GAMMAS] steps=$SAMPLING_STEPS GPU=$CUDA_VISIBLE_DEVICES clf=$CLF_CKPT" | tee -a "$LOG"
for G in $GAMMAS; do
  if [ "$G" = "0" ]; then
    RUN_NAME="noguidance_${TAG}${SUF}"
    GUID_OV=""                                  # gamma=0 = unconditional MDLM (classifier unused)
  else
    RUN_NAME="dcbg_gamma${G}_${TAG}${SUF}"
    GUID_OV="data.label_col=novel data.label_col_pctile=null data.num_classes=2 \
             classifier_backbone=dit classifier_model=tiny-classifier \
             guidance=cbg guidance.method=cbg guidance.gamma=$G guidance.condition=1 \
             guidance.classifier_time_conditioning=True \
             guidance.classifier_checkpoint_path=$CLF_CKPT"
  fi
  CSV=$OUT_DIR/${RUN_NAME}_results.csv
  if [ -f "$CSV" ]; then
    echo "[$(date '+%F %T')] [skip] gamma=$G (exists: $CSV)" | tee -a "$LOG"
    continue
  fi
  echo "[$(date '+%F %T')] === gamma=$G steps=$SAMPLING_STEPS (N=$((NUM_BATCHES*BATCH_SIZE))) GPU=$CUDA_VISIBLE_DEVICES ===" | tee -a "$LOG"
  $PY -u $EVAL \
    data=qm9 model=small backbone=dit model.length=32 \
    parameterization=subs diffusion=absorbing_state time_conditioning=False T=0 \
    training.guidance=null \
    eval.checkpoint_path=$DIFF_CKPT \
    sampling.steps=$SAMPLING_STEPS \
    sampling.num_sample_batches=$NUM_BATCHES \
    loader.eval_global_batch_size=$BATCH_SIZE \
    eval.generated_samples_path=$OUT_DIR/${RUN_NAME}_samples.json \
    +eval.results_csv_path=$CSV \
    seed=1 \
    hydra.run.dir=$OUT_DIR/novelty_eval_${RUN_NAME} \
    $GUID_OV \
    >> "$LOG" 2>&1
  RC=$?
  if [ $RC -eq 0 ] && [ -f "$CSV" ]; then
    echo "[$(date '+%F %T')] [done] gamma=$G -> $(tail -1 "$CSV")" | tee -a "$LOG"
  else
    echo "[$(date '+%F %T')] [FAIL] gamma=$G (rc=$RC) — see $LOG" | tee -a "$LOG"
  fi
done
echo "[$(date '+%F %T')] sweep_gamma_rollout done (GPU=$CUDA_VISIBLE_DEVICES, steps=$SAMPLING_STEPS)." | tee -a "$LOG"
