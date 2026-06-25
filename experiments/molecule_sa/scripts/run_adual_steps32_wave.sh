#!/bin/bash
# Chained sweep of adual sf+td at steps=32, N=500 on a single GPU.
# Each arg line: "C RHO L0 LMAX SEED"
# Reads configs from a passed-in list file (one config per line).
set -u
GPU=$1
CFG_FILE=$2
RESULT_LOG=$3
export CUDA_VISIBLE_DEVICES=$GPU
OUT_DIR=/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance
SCRIPT=/local/scratch/zhiheng/guidance/experiments/molecule_sa/scripts/sample_sa_dcbg_adaptive_samplefirst_timedep_seed.sh
echo "=== wave start $(date) on GPU$GPU ===" >> "$RESULT_LOG"
while read -r C RHO L0 LMAX SEED; do
  [ -z "${C:-}" ] && continue
  case "$C" in \#*) continue;; esac
  TAG=adual_sf_td
  t0=$(date +%s)
  bash "$SCRIPT" "$C" "$RHO" "$L0" "$LMAX" 125 4 3.0 "$TAG" 32 "$SEED" > /local/scratch/zhiheng/guidance/logs/adual32_C${C}_rho${RHO}_l0${L0}_lmax${LMAX}_s${SEED}.log 2>&1
  rc=$?
  t1=$(date +%s)
  STEPS_SUFFIX="_steps32"
  SEED_SUFFIX=""; [ "$SEED" != "1" ] && SEED_SUFFIX="_seed${SEED}"
  RUN_NAME=mdlm_dcbg_sa_${TAG}_C${C}_rho${RHO}_l0${L0}_lmax${LMAX}_trainTau3.0${STEPS_SUFFIX}${SEED_SUFFIX}
  CSV=$OUT_DIR/${RUN_NAME}_results.csv
  if [ $rc -eq 0 ] && [ -e "$CSV" ]; then
    line=$(awk -F, 'NR==2{printf "valid=%s/500 viol3.0=%.4f viol3.5=%.4f",$5,$10,$11}' "$CSV")
    echo "C=$C rho=$RHO l0=$L0 lmax=$LMAX seed=$SEED -> $line  [$((t1-t0))s]" >> "$RESULT_LOG"
  else
    echo "C=$C rho=$RHO l0=$L0 lmax=$LMAX seed=$SEED -> FAILED rc=$rc [$((t1-t0))s]" >> "$RESULT_LOG"
  fi
done < "$CFG_FILE"
echo "=== wave done $(date) ===" >> "$RESULT_LOG"
