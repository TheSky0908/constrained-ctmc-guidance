#!/bin/bash
# Seed sweep on the best config (J4/n1/C95/rho0.15/l0=2/lmax2.9 = 4.59% @ seed1) — user-authorized
# last resort after params exhausted. Seeds 2-8 (<=8 total incl seed1). Records each result.
# Usage: run_seed_sweep.sh GPU SEED1 [SEED2 ...]
GPU=${1:-2}; shift
cd /local/scratch/zhiheng/guidance
SUMMARY=experiments/molecule_sa/il_search128_seed_sweep_lmax2p9.txt
C=0.05129; RHO=0.15; L0=2.0; LMAX=2.9; J=4; N=1

for SEED in "$@"; do
  TAG="seed_J4n1_C95_lmax2p9_s${SEED}"
  OUT=outputs/qm9/il_search128/${TAG}/results.csv
  if [ -f "$OUT" ]; then echo "[seed] SKIP $TAG"; continue; fi
  echo "[seed] START $TAG  $(date +%H:%M:%S)"
  bash experiments/molecule_sa/scripts/run_il_search_steps.sh $GPU $C $RHO $L0 $LMAX $J $N $SEED $TAG 128 125
  if [ -f "$OUT" ]; then
    line=$(tail -1 "$OUT"); viol=$(echo "$line" | cut -d, -f10); valid=$(echo "$line" | cut -d, -f5)
    echo "$TAG  lmax=$LMAX J=$J n=$N seed=$SEED  viol3.0=$viol valid=$valid" | tee -a $SUMMARY
  else
    echo "$TAG  FAILED seed=$SEED" | tee -a $SUMMARY
  fi
done
echo "[seed] DONE $(date +%H:%M:%S)"
