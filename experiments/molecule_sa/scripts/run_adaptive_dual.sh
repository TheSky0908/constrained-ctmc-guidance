#!/bin/bash
set -e
cd /local/scratch/zhiheng/guidance
export CUDA_VISIBLE_DEVICES=1
mkdir -p logs

N_BATCHES=125
BS=4
TAU=3.0
TAG=n500

# C ρ λ₀ λmax
declare -a RUNS=(
  "0.2231 0.5 0.0 50.0"
  "0.5108 0.5 0.0 50.0"
  "0.2231 1.0 0.0 50.0"
)

for r in "${RUNS[@]}"; do
  set -- $r
  C=$1; RHO=$2; L0=$3; LMAX=$4
  LOG=logs/sa_dcbg_adual_${TAG}_C${C}_rho${RHO}_trainTau${TAU}.log
  echo "[$(date)] === adaptive_dual C=$C ρ=$RHO -> $LOG ==="
  bash sample_sa_dcbg_adaptive.sh $C $RHO $L0 $LMAX $N_BATCHES $BS $TAU $TAG > $LOG 2>&1
  grep -E "adaptive_dual\] step|Valid               :|Viol|Novel" $LOG | head -20
done

echo "[$(date)] all 3 adaptive_dual runs done"
