#!/bin/bash
# Sweep: adaptive-dual (sample_first) + TIME-DEPENDENT classifier on SA / QM9.
#   time_conditioning=False (base MDLM) + classifier_time_conditioning=True
#   gamma_adual_update_order=sample_first  (paper Algorithm 1)
#   steps=128, N=500, λ₀=2, λ_max=5
#   C ∈ {-log0.8, -log0.99, -log1.01} = {0.2231, 0.01005, -0.00995}
#   ρ ∈ {0.1, 0.2, 0.5}
# = 9 configs, run sequentially on ONE GPU. Each sa_eval run records its own
# pure sampling wall-clock (sampling_seconds) into the results CSV.
#
# Usage: CUDA_VISIBLE_DEVICES=3 bash run_adual_samplefirst_timedep_sweep.sh

set -e
cd /local/scratch/zhiheng/guidance

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
SCRIPT=experiments/molecule_sa/scripts/sample_sa_dcbg_adaptive_samplefirst_timedep.sh

LAMBDA0=2.0
LMAX=5.0
NB=125     # 125 batches × 4 = N=500
BS=4
TAU=3.0
TAG=adual_sf_td
STEPS=128

echo "[$(date)] === adual sample_first + time-dep sweep START (GPU=$CUDA_VISIBLE_DEVICES) ==="
SWEEP_T0=$(date +%s)

for C in 0.2231 0.01005 -0.00995; do
  for RHO in 0.1 0.2 0.5; do
    echo "[$(date)] --- run C=$C ρ=$RHO λ₀=$LAMBDA0 λmax=$LMAX steps=$STEPS ---"
    RUN_T0=$(date +%s)
    bash "$SCRIPT" "$C" "$RHO" "$LAMBDA0" "$LMAX" "$NB" "$BS" "$TAU" "$TAG" "$STEPS"
    RUN_T1=$(date +%s)
    echo "[$(date)] --- done C=$C ρ=$RHO : wallclock $((RUN_T1-RUN_T0)) s ---"
  done
done

SWEEP_T1=$(date +%s)
echo "[$(date)] === sweep DONE : total wallclock $((SWEEP_T1-SWEEP_T0)) s ==="
