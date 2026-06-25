#!/bin/bash
# Second sequential queue (GPU 2) for IL+td steps=128 N=500 search, complementary configs.
# Format per line: "C RHO L0 LMAX J N SEED TAG". Skips configs whose results.csv exists.
GPU=${1:-2}
cd /local/scratch/zhiheng/guidance
SUMMARY=experiments/molecule_sa/il_search128_queue_results_g2.txt

# Batch 3 GPU2 = region A (low lmax, n=1/8, J=2/3, C/rho cross), l0=2.
CONFIGS=(
  "0.05129 0.15 2.0 3.0 2 1 1 b3_J2n1_C95_lmax3_r15"
  "0.10536 0.20 2.0 3.0 2 1 1 b3_J2n1_C90_lmax3_r20"
  "0.10536 0.20 2.0 3.0 2 4 1 b3_J2n4_C90_lmax3_r20"
  "0.05129 0.15 2.0 2.5 3 1 1 b3_J3n1_C95_lmax2p5_r15"
  "0.22314 0.10 2.0 4.0 2 1 1 b3_J2n1_C80_lmax4_r10"
  "0.10536 0.20 2.0 5.0 2 8 1 b3_J2n8_C90_lmax5_r20"
)

for cfg in "${CONFIGS[@]}"; do
  read C RHO L0 LMAX J N SEED TAG <<< "$cfg"
  OUT=outputs/qm9/il_search128/${TAG}/results.csv
  if [ -f "$OUT" ]; then
    echo "[queue-g2] SKIP $TAG (results exist)"; continue
  fi
  echo "[queue-g2] START $TAG  $(date +%H:%M:%S)"
  bash experiments/molecule_sa/scripts/run_il_search_steps.sh $GPU $C $RHO $L0 $LMAX $J $N $SEED $TAG 128 125
  if [ -f "$OUT" ]; then
    line=$(tail -1 "$OUT")
    viol=$(echo "$line" | cut -d, -f10); valid=$(echo "$line" | cut -d, -f5)
    echo "$TAG  C=$C rho=$RHO l0=$L0 lmax=$LMAX J=$J n=$N seed=$SEED  viol3.0=$viol valid=$valid" | tee -a $SUMMARY
  else
    echo "$TAG  FAILED (no results.csv) C=$C rho=$RHO lmax=$LMAX J=$J n=$N seed=$SEED" | tee -a $SUMMARY
  fi
done
echo "[queue-g2] DONE $(date +%H:%M:%S)"
