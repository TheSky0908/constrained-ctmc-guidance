#!/bin/bash
set -e
cd /local/scratch/zhiheng/guidance
export CUDA_VISIBLE_DEVICES=1
mkdir -p logs

# N=500 = 125 batches x 4
N_BATCHES=125
BS=4
TAU=3.0
TAG=n500

declare -a RUNS=(
  "linear_increasing 1.0 5.0"
  "linear_increasing 1.0 8.0"
  "quadratic_increasing 1.0 5.0"
)

for r in "${RUNS[@]}"; do
  set -- $r
  SCHED=$1; GMIN=$2; GMAX=$3
  LOG=logs/sa_dcbg_${TAG}_${SCHED}_gmin${GMIN}_gmax${GMAX}_trainTau${TAU}.log
  echo "[$(date)] === running $SCHED gmin=$GMIN gmax=$GMAX -> $LOG ==="
  bash sample_sa_dcbg_schedule.sh $SCHED $GMIN $GMAX $N_BATCHES $BS $TAU $TAG > $LOG 2>&1
  tail -25 $LOG
done

echo "[$(date)] all 3 schedules done"
