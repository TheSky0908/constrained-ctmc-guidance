#!/bin/bash
# Batch 6: push toward <6.0% around the 6.38% winner (J4n1/C95/lmax2.5/r15, region A, n=1).
# J lever (J2->J3->J4 at lmax2.5 = 7.91->7.16->6.38) -> try J=6 + J5 lmax variants + rho/C tweaks.
# Usage: run_queue128_b6.sh GPU
GPU=${1:-2}
cd /local/scratch/zhiheng/guidance
SUMMARY=experiments/molecule_sa/il_search128_queue_results_b6.txt

CONFIGS=(
  "0.05129 0.15 2.0 2.5  6 1 1 r6_J6n1_C95_lmax2p5"
  "0.05129 0.15 2.0 2.25 5 1 1 r6_J5n1_C95_lmax2p25"
  "0.05129 0.15 2.0 2.75 5 1 1 r6_J5n1_C95_lmax2p75"
  "0.05129 0.20 2.0 2.5  4 1 1 r6_J4n1_C95_lmax2p5_r20"
  "0.05129 0.10 2.0 2.5  4 1 1 r6_J4n1_C95_lmax2p5_r10"
  "0.01005 0.15 2.0 2.5  4 1 1 r6_J4n1_C99_lmax2p5"
)

for cfg in "${CONFIGS[@]}"; do
  read C RHO L0 LMAX J N SEED TAG <<< "$cfg"
  OUT=outputs/qm9/il_search128/${TAG}/results.csv
  if [ -f "$OUT" ]; then echo "[queue-b6] SKIP $TAG"; continue; fi
  echo "[queue-b6] START $TAG  $(date +%H:%M:%S)"
  bash experiments/molecule_sa/scripts/run_il_search_steps.sh $GPU $C $RHO $L0 $LMAX $J $N $SEED $TAG 128 125
  if [ -f "$OUT" ]; then
    line=$(tail -1 "$OUT"); viol=$(echo "$line" | cut -d, -f10); valid=$(echo "$line" | cut -d, -f5)
    echo "$TAG  C=$C rho=$RHO l0=$L0 lmax=$LMAX J=$J n=$N seed=$SEED  viol3.0=$viol valid=$valid" | tee -a $SUMMARY
  else
    echo "$TAG  FAILED C=$C rho=$RHO lmax=$LMAX J=$J n=$N seed=$SEED" | tee -a $SUMMARY
  fi
done
echo "[queue-b6] DONE $(date +%H:%M:%S)"
