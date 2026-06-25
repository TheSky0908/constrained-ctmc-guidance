#!/bin/bash
# Overnight driver: wait for the novelty classifier training to finish, then
# run the baseline novelty sampling points (no-guidance + fixed gamma {1,3,5}).
# Chained so the baselines start the moment GPU 1 frees up.
#
#   bash run_overnight.sh   (launch in background; logs to logs/run_overnight.log)

set -u
cd /local/scratch/zhiheng/guidance

LOGDIR=/local/scratch/zhiheng/guidance/experiments/molecule_novelty/logs
CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/classifier/novelty_absorbing_state_T-0/checkpoints/best.ckpt
MARK () { echo "[$(date '+%F %T')] $*"; }

MARK "overnight driver start; waiting for classifier training to finish..."

# ── 1. wait for training process to exit (safety cap 4h) ───────────────────────
WAITED=0
CAP=14400
while pgrep -f "mode=train_classifier" >/dev/null; do
  sleep 60
  WAITED=$((WAITED + 60))
  if [ $((WAITED % 600)) -eq 0 ]; then
    LASTCE=$(grep -oE "val/cross_entropy' reached [0-9.]+" "$LOGDIR/novelty_classifier.log" 2>/dev/null | tail -1)
    MARK "still training (${WAITED}s elapsed) | $LASTCE"
  fi
  if [ "$WAITED" -ge "$CAP" ]; then
    MARK "WARNING: training still alive after ${CAP}s cap — aborting to avoid GPU contention."
    exit 1
  fi
done
MARK "training process exited (waited ${WAITED}s)."

# ── 2. confirm best.ckpt ───────────────────────────────────────────────────────
if [ ! -f "$CKPT" ]; then
  MARK "ERROR: best.ckpt not found at $CKPT — running no-guidance baseline only."
else
  MARK "best.ckpt present: $(ls -la "$CKPT" | awk '{print $5" bytes"}')"
fi
FINALCE=$(grep -oE "val/cross_entropy' reached [0-9.]+" "$LOGDIR/novelty_classifier.log" 2>/dev/null | tail -1)
MARK "final $FINALCE"

# ── 3. run baselines on GPU 1 (no-guidance + gamma {1,3,5}, N=500) ─────────────
MARK "launching baselines (N=500, steps=128)..."
export CUDA_VISIBLE_DEVICES=1
bash experiments/molecule_novelty/scripts/sample_novelty_baselines.sh 125 4 128 \
  >> "$LOGDIR/sample_novelty_baselines.log" 2>&1
RC=$?
MARK "baselines finished (rc=$RC)."

# ── 4. summary ─────────────────────────────────────────────────────────────────
MARK "=== RESULTS SUMMARY (valid&novel is the headline KPI) ==="
RESDIR=/local/scratch/zhiheng/guidance/experiments/molecule_novelty/results
for csv in "$RESDIR"/noguidance_results.csv "$RESDIR"/dcbg_gamma1_results.csv \
           "$RESDIR"/dcbg_gamma3_results.csv "$RESDIR"/dcbg_gamma5_results.csv; do
  if [ -f "$csv" ]; then
    MARK "--- $(basename "$csv") ---"
    cat "$csv"
  fi
done
MARK "CDD reference (No Guidance, valid&novel): MDLM 271 | UDLM 345 | CDD 511"
MARK "overnight driver done."
