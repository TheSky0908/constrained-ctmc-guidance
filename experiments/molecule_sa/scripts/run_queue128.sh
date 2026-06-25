#!/bin/bash
# Sequential queue driver for IL+td steps=128 N=500 search on a single GPU.
# Edit the CONFIGS array (one line per run: "C RHO L0 LMAX J N SEED TAG").
# Skips any config whose results.csv already exists. Appends a summary line per run.
GPU=${1:-7}
cd /local/scratch/zhiheng/guidance
SUMMARY=experiments/molecule_sa/il_search128_queue_results.txt

# Batch 5 GPU7 = region A high-J (around new best J3n1/C95/lmax2.5/r15 = 7.16%), n=1, l0=2.
# J2->J3 at lmax2.5 dropped 7.91->7.16 -> push J=4/5 + lmax fine grid.
# Format: C RHO L0 LMAX J N SEED TAG
CONFIGS=(
  "0.05129 0.15 2.0 2.5  4 1 1 r5_J4n1_C95_lmax2p5"
  "0.05129 0.15 2.0 2.25 4 1 1 r5_J4n1_C95_lmax2p25"
  "0.05129 0.15 2.0 2.5  5 1 1 r5_J5n1_C95_lmax2p5"
  "0.05129 0.15 2.0 2.25 3 1 1 r5_J3n1_C95_lmax2p25"
  "0.05129 0.15 2.0 2.75 4 1 1 r5_J4n1_C95_lmax2p75"
  "0.10536 0.20 2.0 2.5  4 1 1 r5_J4n1_C90_lmax2p5"
)

for cfg in "${CONFIGS[@]}"; do
  read C RHO L0 LMAX J N SEED TAG <<< "$cfg"
  OUT=outputs/qm9/il_search128/${TAG}/results.csv
  if [ -f "$OUT" ]; then
    echo "[queue] SKIP $TAG (results exist)"; continue
  fi
  echo "[queue] START $TAG  $(date +%H:%M:%S)"
  bash experiments/molecule_sa/scripts/run_il_search_steps.sh $GPU $C $RHO $L0 $LMAX $J $N $SEED $TAG 128 125
  if [ -f "$OUT" ]; then
    line=$(tail -1 "$OUT")
    viol=$(echo "$line" | cut -d, -f10); valid=$(echo "$line" | cut -d, -f5)
    echo "$TAG  C=$C rho=$RHO l0=$L0 lmax=$LMAX J=$J n=$N seed=$SEED  viol3.0=$viol valid=$valid" | tee -a $SUMMARY
  else
    echo "$TAG  FAILED (no results.csv) C=$C rho=$RHO lmax=$LMAX J=$J n=$N seed=$SEED" | tee -a $SUMMARY
  fi
done
echo "[queue] DONE $(date +%H:%M:%S)"
