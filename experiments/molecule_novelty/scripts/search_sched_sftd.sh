#!/bin/bash
# Deterministic time-varying gamma SCHEDULE search (td/rollout classifier) on the
# NOVELTY task. Tests whether a ramped gamma(t) can break the Viol=0 <-> validity
# tension that flat/adual guidance hits at steps=64.
#
# Schedules (see diffusion._compute_gamma_t): t_norm: 1=fully noisy(start), 0=clean(end)
#   linear_increasing  : gmin@start -> gmax@end   (low early, HIGH late)
#   linear_decreasing  : gmax@start -> gmin@end   (HIGH early, low late)
#   quadratic_increasing, cosine_increasing, step
#
# Combo format: "SCHED:GMIN:GMAX:SEED:STEPS"   (STEPS optional, default 64)
# Usage: CUDA_VISIBLE_DEVICES=2 bash search_sched_sftd.sh "linear_decreasing:1.5:4.0:1" ...

set -uo pipefail
cd /local/scratch/zhiheng/guidance

DEFAULT_STEPS=64
NUM_BATCHES=125
BATCH_SIZE=4

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export HF_HOME=/local/scratch/zhiheng/guidance/.hf_cache
export PYTHONPATH=/local/scratch/zhiheng/guidance:/local/scratch/zhiheng/guidance/guidance_eval

PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
DIFF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt
CLF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/classifier/novelty_rollout_timedep/checkpoints/best.ckpt
OUT_DIR=/local/scratch/zhiheng/guidance/experiments/molecule_novelty/results
SEARCH_DIR=$OUT_DIR/sched_sftd_search
EVAL=experiments/molecule_novelty/novelty_eval.py
LOGDIR=/local/scratch/zhiheng/guidance/experiments/molecule_novelty/logs
mkdir -p "$SEARCH_DIR" "$LOGDIR"
HDR="sched,gmin,gmax,seed,steps,seconds,valid,unique,valid_novel,viol,novel_rate,qed,run_name"

for p in "$DIFF_CKPT" "$CLF_CKPT"; do
  [ -f "$p" ] || { echo "[search_sched_sftd] MISSING CKPT: $p" >&2; exit 1; }
done

for combo in "$@"; do
  IFS=':' read -r SCHED GMIN GMAX SEED STEPS <<< "$combo"
  SEED=${SEED:-1}
  SAMPLING_STEPS=${STEPS:-$DEFAULT_STEPS}
  MASTER=$OUT_DIR/sched_sftd_search_steps${SAMPLING_STEPS}.csv
  [ -f "$MASTER" ] || echo "$HDR" > "$MASTER"
  RUN_NAME=sched_${SCHED}_gmin${GMIN}_gmax${GMAX}_seed${SEED}_steps${SAMPLING_STEPS}
  CSV=$SEARCH_DIR/${RUN_NAME}_results.csv
  LOG=$LOGDIR/${RUN_NAME}.log
  if [ -f "$CSV" ]; then echo "[$(date '+%F %T')] [skip] $RUN_NAME (exists)"; continue; fi
  echo "[$(date '+%F %T')] === $SCHED gmin=$GMIN gmax=$GMAX seed=$SEED steps=$SAMPLING_STEPS GPU=$CUDA_VISIBLE_DEVICES ==="
  T0=$(date +%s)
  $PY -u $EVAL \
    data=qm9 model=small backbone=dit model.length=32 \
    parameterization=subs diffusion=absorbing_state time_conditioning=False T=0 \
    training.guidance=null \
    data.label_col=novel data.label_col_pctile=null data.num_classes=2 \
    classifier_backbone=dit classifier_model=tiny-classifier \
    eval.checkpoint_path=$DIFF_CKPT \
    guidance=cbg guidance.method=cbg guidance.gamma=1.0 \
    guidance.gamma_schedule=$SCHED \
    guidance.gamma_min=$GMIN guidance.gamma_max=$GMAX \
    guidance.classifier_time_conditioning=True \
    guidance.condition=1 guidance.classifier_checkpoint_path=$CLF_CKPT \
    sampling.steps=$SAMPLING_STEPS \
    sampling.num_sample_batches=$NUM_BATCHES \
    loader.eval_global_batch_size=$BATCH_SIZE \
    eval.generated_samples_path=$SEARCH_DIR/${RUN_NAME}_samples.json \
    +eval.results_csv_path=$CSV \
    seed=$SEED \
    hydra.run.dir=$SEARCH_DIR/eval_${RUN_NAME} \
    >> "$LOG" 2>&1
  RC=$?
  SEC=$(( $(date +%s) - T0 ))
  if [ $RC -eq 0 ] && [ -f "$CSV" ]; then
    L=$(tail -1 "$CSV")
    VALID=$(echo "$L" | cut -d, -f9); UNIQ=$(echo "$L" | cut -d, -f10)
    VN=$(echo "$L" | cut -d, -f11); QED=$(echo "$L" | cut -d, -f13)
    NR=$(echo "$L" | cut -d, -f14); VIOL=$(echo "$L" | cut -d, -f15)
    echo "$SCHED,$GMIN,$GMAX,$SEED,$SAMPLING_STEPS,$SEC,$VALID,$UNIQ,$VN,$VIOL,$NR,$QED,$RUN_NAME" >> "$MASTER"
    echo "[$(date '+%F %T')] [done] ${SEC}s valid=$VALID viol=$VIOL vn=$VN -> $MASTER"
  else
    echo "$SCHED,$GMIN,$GMAX,$SEED,$SAMPLING_STEPS,$SEC,FAIL,,,,,,$RUN_NAME" >> "$MASTER"
    echo "[$(date '+%F %T')] [FAIL] rc=$RC ${SEC}s — see $LOG"
  fi
done
echo "[$(date '+%F %T')] sched search batch done (GPU=$CUDA_VISIBLE_DEVICES)."
