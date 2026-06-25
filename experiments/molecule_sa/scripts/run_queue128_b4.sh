#!/bin/bash
# Batch 4: looser-C + lambda_max fine grid around the J4n4/C90/lmax5 winning point (=7.55%).
# l0=2 fixed, seed=1. Looser C (-log0.85=0.16252, -log0.80=0.22314) lets lambda decay after
# the classifier clears its higher -C target -> less over-constraint.
# Usage: run_queue128_b4.sh GPU
GPU=${1:-7}
cd /local/scratch/zhiheng/guidance
SUMMARY=experiments/molecule_sa/il_search128_queue_results_b4.txt

# J=4 is best J so far (J4n4=7.55% < J3n4=9.05%). Push higher J + looser C.
CONFIGS=(
  "0.10536 0.20 2.0 5.0 5 4 1 b4_J5n4_C90_lmax5"
  "0.16252 0.20 2.0 5.0 4 4 1 b4_J4n4_C85_lmax5"
  "0.22314 0.20 2.0 5.0 4 4 1 b4_J4n4_C80_lmax5"
  "0.10536 0.20 2.0 5.0 6 4 1 b4_J6n4_C90_lmax5"
  "0.16252 0.20 2.0 5.0 5 4 1 b4_J5n4_C85_lmax5"
  "0.10536 0.20 2.0 4.0 5 4 1 b4_J5n4_C90_lmax4"
)

for cfg in "${CONFIGS[@]}"; do
  read C RHO L0 LMAX J N SEED TAG <<< "$cfg"
  OUT=outputs/qm9/il_search128/${TAG}/results.csv
  if [ -f "$OUT" ]; then
    echo "[queue-b4] SKIP $TAG (results exist)"; continue
  fi
  echo "[queue-b4] START $TAG  $(date +%H:%M:%S)"
  bash experiments/molecule_sa/scripts/run_il_search_steps.sh $GPU $C $RHO $L0 $LMAX $J $N $SEED $TAG 128 125
  if [ -f "$OUT" ]; then
    line=$(tail -1 "$OUT")
    viol=$(echo "$line" | cut -d, -f10); valid=$(echo "$line" | cut -d, -f5)
    echo "$TAG  C=$C rho=$RHO l0=$L0 lmax=$LMAX J=$J n=$N seed=$SEED  viol3.0=$viol valid=$valid" | tee -a $SUMMARY
  else
    echo "$TAG  FAILED C=$C rho=$RHO lmax=$LMAX J=$J n=$N seed=$SEED" | tee -a $SUMMARY
  fi
done
echo "[queue-b4] DONE $(date +%H:%M:%S)"
