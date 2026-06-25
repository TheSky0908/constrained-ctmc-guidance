#!/bin/bash
# Per-gamma TIMED constant-gamma D-CBG novelty sweep (rollout/timedep classifier).
# Same command as sweep_gamma_rollout.sh, but runs SINGLE-CARD sequentially and
# records EACH gamma's own wall-clock seconds into a timing CSV, so every
# parameter row carries its own Time(s) (mirrors the molecule_sa K=32 table).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash time_gamma_rollout.sh ["0 1 .. 6"] [STEPS]
set -uo pipefail
cd /local/scratch/zhiheng/guidance

GAMMAS=${1:-"0 1 2 3 4 5 6"}
SAMPLING_STEPS=${2:-64}
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
OUT_DIR=/local/scratch/zhiheng/guidance/experiments/molecule_novelty/results/timing
EVAL=experiments/molecule_novelty/novelty_eval.py
LOGDIR=/local/scratch/zhiheng/guidance/experiments/molecule_novelty/logs
mkdir -p "$OUT_DIR" "$LOGDIR"
TIMING_CSV=$OUT_DIR/timing_${TAG}${SUF}_gpu${CUDA_VISIBLE_DEVICES}.csv
LOG=$LOGDIR/time_gamma_${TAG}${SUF}_gpu${CUDA_VISIBLE_DEVICES}.log

for p in "$DIFF_CKPT" "$CLF_CKPT"; do
  [ -f "$p" ] || { echo "[time_gamma_rollout] MISSING CKPT: $p" >&2; exit 1; }
done

echo "gamma,steps,seconds,valid,viol,valid_novel" > "$TIMING_CSV"
echo "[$(date '+%F %T')] time_gamma_rollout start: gammas=[$GAMMAS] steps=$SAMPLING_STEPS GPU=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"
for G in $GAMMAS; do
  if [ "$G" = "0" ]; then
    RUN_NAME="noguidance_${TAG}${SUF}"; GUID_OV=""
  else
    RUN_NAME="dcbg_gamma${G}_${TAG}${SUF}"
    GUID_OV="data.label_col=novel data.label_col_pctile=null data.num_classes=2 \
             classifier_backbone=dit classifier_model=tiny-classifier \
             guidance=cbg guidance.method=cbg guidance.gamma=$G guidance.condition=1 \
             guidance.classifier_time_conditioning=True \
             guidance.classifier_checkpoint_path=$CLF_CKPT"
  fi
  CSV=$OUT_DIR/${RUN_NAME}_results.csv
  echo "[$(date '+%F %T')] === gamma=$G steps=$SAMPLING_STEPS GPU=$CUDA_VISIBLE_DEVICES ===" | tee -a "$LOG"
  T0=$(date +%s)
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
  SEC=$(( $(date +%s) - T0 ))
  if [ $RC -eq 0 ] && [ -f "$CSV" ]; then
    LAST=$(tail -1 "$CSV")
    VALID=$(echo "$LAST" | cut -d, -f9)
    VN=$(echo "$LAST" | cut -d, -f11)
    VIOL=$(echo "$LAST" | cut -d, -f15)
    echo "$G,$SAMPLING_STEPS,$SEC,$VALID,$VIOL,$VN" >> "$TIMING_CSV"
    echo "[$(date '+%F %T')] [done] gamma=$G time=${SEC}s valid=$VALID viol=$VIOL vn=$VN" | tee -a "$LOG"
  else
    echo "$G,$SAMPLING_STEPS,$SEC,FAIL,," >> "$TIMING_CSV"
    echo "[$(date '+%F %T')] [FAIL] gamma=$G (rc=$RC) time=${SEC}s — see $LOG" | tee -a "$LOG"
  fi
done
echo "[$(date '+%F %T')] time_gamma_rollout done -> $TIMING_CSV" | tee -a "$LOG"
column -t -s, "$TIMING_CSV" | tee -a "$LOG"
