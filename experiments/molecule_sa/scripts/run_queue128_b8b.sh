#!/bin/bash
# Batch 8b (GPU7, parallel to b8 on GPU2): bracket the HIGH lambda_max end at J=4 to find where
# Viol rebounds (lmax 2.25/2.5/2.75 = 8.01/6.38/5.09, dropping). C95/rho0.15/l0=2/n=1, seed=1.
GPU=${1:-7}
cd /local/scratch/zhiheng/guidance
SUMMARY=experiments/molecule_sa/il_search128_queue_results_b8b.txt

CONFIGS=(
  "0.05129 0.15 2.0 3.75 4 1 1 r8_J4n1_C95_lmax3p75"
  "0.05129 0.15 2.0 4.5  4 1 1 r8_J4n1_C95_lmax4p5"
  "0.05129 0.15 2.0 5.0  4 1 1 r8_J4n1_C95_lmax5p0"
  "0.05129 0.15 2.0 6.0  4 1 1 r8_J4n1_C95_lmax6p0"
  "0.05129 0.15 2.0 2.9  4 1 1 r8_J4n1_C95_lmax2p9"
)

for cfg in "${CONFIGS[@]}"; do
  read C RHO L0 LMAX J N SEED TAG <<< "$cfg"
  OUT=outputs/qm9/il_search128/${TAG}/results.csv
  if [ -f "$OUT" ]; then echo "[queue-b8b] SKIP $TAG"; continue; fi
  echo "[queue-b8b] START $TAG  $(date +%H:%M:%S)"
  bash experiments/molecule_sa/scripts/run_il_search_steps.sh $GPU $C $RHO $L0 $LMAX $J $N $SEED $TAG 128 125
  if [ -f "$OUT" ]; then
    line=$(tail -1 "$OUT"); viol=$(echo "$line" | cut -d, -f10); valid=$(echo "$line" | cut -d, -f5)
    echo "$TAG  C=$C rho=$RHO l0=$L0 lmax=$LMAX J=$J n=$N seed=$SEED  viol3.0=$viol valid=$valid" | tee -a $SUMMARY
  else
    echo "$TAG  FAILED C=$C rho=$RHO lmax=$LMAX J=$J n=$N seed=$SEED" | tee -a $SUMMARY
  fi
done
echo "[queue-b8b] DONE $(date +%H:%M:%S)"
