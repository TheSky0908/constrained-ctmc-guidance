#!/bin/bash
# steps=256 sweep with the TIME-DEPENDENT classifier, N=500, train τ=3.0:
#   (1) constant γ=3 (CBG)
#   (2) adaptive-dual sample_first: C ∈ {−log0.8, −log0.99}={0.2231,0.01005}, ρ ∈ {0.1,0.2}, λ₀=2, λ_max=5
# = 5 runs, sequential on ONE GPU. Each records its own sampling_seconds in the CSV.
#
# Usage: CUDA_VISIBLE_DEVICES=2 bash run_steps256_timedep_sweep.sh

set -e
cd /local/scratch/zhiheng/guidance
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

CONST=experiments/molecule_sa/scripts/sample_sa_dcbg_timedep.sh
ADUAL=experiments/molecule_sa/scripts/sample_sa_dcbg_adaptive_samplefirst_timedep.sh
STEPS=256

echo "[$(date)] === steps=256 time-dep sweep START (GPU=$CUDA_VISIBLE_DEVICES) ==="
T0=$(date +%s)

# (1) constant γ=3
echo "[$(date)] --- run constant γ=3, steps=256 ---"
R0=$(date +%s); bash "$CONST" 3 125 4 3.0 "$STEPS"; R1=$(date +%s)
echo "[$(date)] --- done constant γ=3 : wallclock $((R1-R0)) s ---"

# (2) adaptive-dual sample_first, λ₀=2, λ_max=5
for C in 0.2231 0.01005; do
  for RHO in 0.1 0.2; do
    echo "[$(date)] --- run adual C=$C ρ=$RHO λ₀=2 λmax=5, steps=256 ---"
    R0=$(date +%s); bash "$ADUAL" "$C" "$RHO" 2.0 5.0 125 4 3.0 adual_sf_td "$STEPS"; R1=$(date +%s)
    echo "[$(date)] --- done adual C=$C ρ=$RHO : wallclock $((R1-R0)) s ---"
  done
done

T1=$(date +%s)
echo "[$(date)] === steps=256 sweep DONE : total wallclock $((T1-T0)) s ==="
