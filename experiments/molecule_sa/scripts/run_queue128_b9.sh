#!/bin/bash
# Batch 9 (GPU7): fine grid around the lambda_max minimum (~3.0, =4.91%; U-shape, 3.75 rebounds to 7%).
# J4/C95/rho0.15/l0=2/n=1, seed=1. Target <4.5%.
GPU=${1:-7}
cd /local/scratch/zhiheng/guidance
SUMMARY=experiments/molecule_sa/il_search128_queue_results_b9.txt

CONFIGS=(
  "0.05129 0.15 2.0 2.9  4 1 1 r9_J4n1_C95_lmax2p9"
  "0.05129 0.15 2.0 2.95 4 1 1 r9_J4n1_C95_lmax2p95"
  "0.05129 0.15 2.0 3.05 4 1 1 r9_J4n1_C95_lmax3p05"
  "0.05129 0.15 2.0 3.1  4 1 1 r9_J4n1_C95_lmax3p1"
  "0.05129 0.15 2.0 2.85 4 1 1 r9_J4n1_C95_lmax2p85"
  "0.05129 0.15 2.0 3.15 4 1 1 r9_J4n1_C95_lmax3p15"
)

for cfg in "${CONFIGS[@]}"; do
  read C RHO L0 LMAX J N SEED TAG <<< "$cfg"
  OUT=outputs/qm9/il_search128/${TAG}/results.csv
  if [ -f "$OUT" ]; then echo "[queue-b9] SKIP $TAG"; continue; fi
  echo "[queue-b9] START $TAG  $(date +%H:%M:%S)"
  bash experiments/molecule_sa/scripts/run_il_search_steps.sh $GPU $C $RHO $L0 $LMAX $J $N $SEED $TAG 128 125
  if [ -f "$OUT" ]; then
    line=$(tail -1 "$OUT"); viol=$(echo "$line" | cut -d, -f10); valid=$(echo "$line" | cut -d, -f5)
    echo "$TAG  C=$C rho=$RHO l0=$L0 lmax=$LMAX J=$J n=$N seed=$SEED  viol3.0=$viol valid=$valid" | tee -a $SUMMARY
  else
    echo "$TAG  FAILED C=$C rho=$RHO lmax=$LMAX J=$J n=$N seed=$SEED" | tee -a $SUMMARY
  fi
done
echo "[queue-b9] DONE $(date +%H:%M:%S)"
