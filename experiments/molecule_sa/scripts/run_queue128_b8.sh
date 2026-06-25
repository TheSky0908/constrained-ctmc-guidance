#!/bin/bash
# Batch 8: push lambda_max UP at J=4 (lmax 2.25/2.5/2.75 = 8.01/6.38/5.09, still dropping).
# Target <5.0%. C95/rho0.15/l0=2/n=1 fixed (C is clip-dominated; lmax is the decisive lever).
# Usage: run_queue128_b8.sh GPU
GPU=${1:-2}
cd /local/scratch/zhiheng/guidance
SUMMARY=experiments/molecule_sa/il_search128_queue_results_b8.txt

CONFIGS=(
  "0.05129 0.15 2.0 3.0  4 1 1 r8_J4n1_C95_lmax3p0"
  "0.05129 0.15 2.0 3.25 4 1 1 r8_J4n1_C95_lmax3p25"
  "0.05129 0.15 2.0 3.5  4 1 1 r8_J4n1_C95_lmax3p5"
  "0.05129 0.15 2.0 4.0  4 1 1 r8_J4n1_C95_lmax4p0"
  "0.05129 0.20 2.0 3.0  4 1 1 r8_J4n1_C95_lmax3p0_r20"
  "0.05129 0.15 2.0 3.25 5 1 1 r8_J5n1_C95_lmax3p25"
)

for cfg in "${CONFIGS[@]}"; do
  read C RHO L0 LMAX J N SEED TAG <<< "$cfg"
  OUT=outputs/qm9/il_search128/${TAG}/results.csv
  if [ -f "$OUT" ]; then echo "[queue-b8] SKIP $TAG"; continue; fi
  echo "[queue-b8] START $TAG  $(date +%H:%M:%S)"
  bash experiments/molecule_sa/scripts/run_il_search_steps.sh $GPU $C $RHO $L0 $LMAX $J $N $SEED $TAG 128 125
  if [ -f "$OUT" ]; then
    line=$(tail -1 "$OUT"); viol=$(echo "$line" | cut -d, -f10); valid=$(echo "$line" | cut -d, -f5)
    echo "$TAG  C=$C rho=$RHO l0=$L0 lmax=$LMAX J=$J n=$N seed=$SEED  viol3.0=$viol valid=$valid" | tee -a $SUMMARY
  else
    echo "$TAG  FAILED C=$C rho=$RHO lmax=$LMAX J=$J n=$N seed=$SEED" | tee -a $SUMMARY
  fi
done
echo "[queue-b8] DONE $(date +%H:%M:%S)"
