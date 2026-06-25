#!/bin/bash
# Batch 7: fine-tune AROUND the J=4 optimum (J4n1/C95/lmax2.5/r15 = 6.38%) toward <6.0%.
# J is peaked at 4 (J2..J5 = 7.91/7.16/6.38/7.51); lmax peaked at 2.5 (2.25=8.01). So sweep C/rho/lmax.
# Usage: run_queue128_b7.sh GPU
GPU=${1:-2}
cd /local/scratch/zhiheng/guidance
SUMMARY=experiments/molecule_sa/il_search128_queue_results_b7.txt

CONFIGS=(
  "0.01005 0.15 2.0 2.5 4 1 1 r7_J4n1_C99_lmax2p5_r15"
  "0.10536 0.15 2.0 2.5 4 1 1 r7_J4n1_C90_lmax2p5_r15"
  "0.05129 0.20 2.0 2.5 4 1 1 r7_J4n1_C95_lmax2p5_r20"
  "0.05129 0.10 2.0 2.5 4 1 1 r7_J4n1_C95_lmax2p5_r10"
  "0.05129 0.15 2.0 2.6 4 1 1 r7_J4n1_C95_lmax2p6_r15"
  "0.16252 0.15 2.0 2.5 4 1 1 r7_J4n1_C85_lmax2p5_r15"
)

for cfg in "${CONFIGS[@]}"; do
  read C RHO L0 LMAX J N SEED TAG <<< "$cfg"
  OUT=outputs/qm9/il_search128/${TAG}/results.csv
  if [ -f "$OUT" ]; then echo "[queue-b7] SKIP $TAG"; continue; fi
  echo "[queue-b7] START $TAG  $(date +%H:%M:%S)"
  bash experiments/molecule_sa/scripts/run_il_search_steps.sh $GPU $C $RHO $L0 $LMAX $J $N $SEED $TAG 128 125
  if [ -f "$OUT" ]; then
    line=$(tail -1 "$OUT"); viol=$(echo "$line" | cut -d, -f10); valid=$(echo "$line" | cut -d, -f5)
    echo "$TAG  C=$C rho=$RHO l0=$L0 lmax=$LMAX J=$J n=$N seed=$SEED  viol3.0=$viol valid=$valid" | tee -a $SUMMARY
  else
    echo "$TAG  FAILED C=$C rho=$RHO lmax=$LMAX J=$J n=$N seed=$SEED" | tee -a $SUMMARY
  fi
done
echo "[queue-b7] DONE $(date +%H:%M:%S)"
