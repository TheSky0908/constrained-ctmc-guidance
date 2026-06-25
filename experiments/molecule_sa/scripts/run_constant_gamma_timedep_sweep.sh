#!/bin/bash
# Constant-γ CBG sweep with the TIME-DEPENDENT classifier.
#   steps=128, N=500, γ ∈ {0, 1, 3, 5, 10}, train τ = 3.0
# Each run records its own pure sampling wall-clock (sampling_seconds) in the CSV.
#
# Usage: CUDA_VISIBLE_DEVICES=3 bash run_constant_gamma_timedep_sweep.sh

set -e
cd /local/scratch/zhiheng/guidance
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
SCRIPT=experiments/molecule_sa/scripts/sample_sa_dcbg_timedep.sh

echo "[$(date)] === constant-γ + time-dep sweep START (GPU=$CUDA_VISIBLE_DEVICES) ==="
T0=$(date +%s)
for G in 0 1 3 5 10; do
  echo "[$(date)] --- run γ=$G ---"
  R0=$(date +%s)
  bash "$SCRIPT" "$G" 125 4 3.0 128
  R1=$(date +%s)
  echo "[$(date)] --- done γ=$G : wallclock $((R1-R0)) s ---"
done
T1=$(date +%s)
echo "[$(date)] === sweep DONE : total wallclock $((T1-T0)) s ==="
