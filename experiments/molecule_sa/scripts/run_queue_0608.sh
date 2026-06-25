#!/bin/bash
# Sequential experiment driver for the 2026-06-08 overnight session.
# Runs a hard-coded queue of adaptive-dual / constant-gamma sampling jobs
# one at a time on a single GPU (GPU 3), logging per-run wall-clock.
#
# Usage: bash run_queue_0608.sh <queue_name>
#   queue_name = phase1 | phase2
#
# All runs: N=500 (125 batches x 4), train tau = eval tau = 3.0, seed=1.

set -u
cd /local/scratch/zhiheng/guidance

export CUDA_VISIBLE_DEVICES=3
SCRIPTS=experiments/molecule_sa/scripts
LOGDIR=experiments/molecule_sa/logs/session_0608
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/timing_summary.txt"

run_ada() {  # C RHO L0 LMAX STEPS label
  local C=$1 RHO=$2 L0=$3 LMAX=$4 STEPS=$5 LABEL=$6
  local log="$LOGDIR/${LABEL}.log"
  echo "[$(date)] >>> START $LABEL : ada C=$C rho=$RHO l0=$L0 lmax=$LMAX steps=$STEPS" | tee -a "$SUMMARY"
  local t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=3 bash "$SCRIPTS/sample_sa_dcbg_adaptive.sh" "$C" "$RHO" "$L0" "$LMAX" 125 4 3.0 n500 "$STEPS" > "$log" 2>&1
  local rc=$?
  local t1=$(date +%s)
  echo "[$(date)] <<< END   $LABEL rc=$rc elapsed=$((t1-t0))s" | tee -a "$SUMMARY"
}

run_const() {  # GAMMA STEPS label
  local GAMMA=$1 STEPS=$2 LABEL=$3
  local log="$LOGDIR/${LABEL}.log"
  echo "[$(date)] >>> START $LABEL : const gamma=$GAMMA steps=$STEPS" | tee -a "$SUMMARY"
  local t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=3 bash "$SCRIPTS/sample_sa_dcbg.sh" "$GAMMA" 125 4 3.0 "$STEPS" > "$log" 2>&1
  local rc=$?
  local t1=$(date +%s)
  echo "[$(date)] <<< END   $LABEL rc=$rc elapsed=$((t1-t0))s" | tee -a "$SUMMARY"
}

QUEUE=${1:-phase1}
echo "==================== QUEUE $QUEUE @ $(date) ====================" | tee -a "$SUMMARY"

if [ "$QUEUE" = "phase1" ]; then
  # steps=256 one-knob-at-a-time sweep around the known-good winner
  # (rho=0.2,l0=2,lmax=5 -> 6.13%). A1 (rho=0.1) gave 7.24% (too slow a ramp at
  # this coarse grid), so we hold rho>=0.2 and vary cap / ramp / warm-start.
  run_ada 0.01005 0.2  2.0 4.0  256 A2_rho0.2_l02_lmax4_steps256   # lower cap
  run_ada 0.01005 0.2  2.0 6.0  256 A3_rho0.2_l02_lmax6_steps256   # higher cap
  run_ada 0.01005 0.3  2.0 5.0  256 A4_rho0.3_l02_lmax5_steps256   # faster ramp
  run_ada 0.01005 0.2  3.0 5.0  256 A5_rho0.2_l03_lmax5_steps256   # stronger warm-start
  echo "[$(date)] PHASE1 COMPLETE" | tee -a "$SUMMARY"

elif [ "$QUEUE" = "phase2" ]; then
  # steps=512: constant gamma=3 baseline first, then adaptive configs.
  run_const 3 512 B0_const_gamma3_steps512
  run_ada 0.01005 0.1  2.0 5.0  512 B1_rho0.1_l02_lmax5_steps512
  run_ada 0.01005 0.2  2.0 5.0  512 B2_rho0.2_l02_lmax5_steps512
  run_ada 0.01005 0.1  2.0 4.0  512 B3_rho0.1_l02_lmax4_steps512
  run_ada 0.01005 0.15 2.0 5.0  512 B4_rho0.15_l02_lmax5_steps512
  echo "[$(date)] PHASE2 COMPLETE" | tee -a "$SUMMARY"

elif [ "$QUEUE" = "round2" ]; then
  # 256-R2: probe cap below 4 (plateau already sits <4 since lmax4==lmax5).
  run_ada 0.01005 0.2 2.0 3.0  256 C2a_rho0.2_l02_lmax3_steps256
  run_ada 0.01005 0.2 2.0 3.5  256 C2b_rho0.2_l02_lmax3.5_steps256
  # 512 line: baseline first, then bracket rho (optimum trends down with finer grid).
  run_const 3 512 B0_const_gamma3_steps512
  run_ada 0.01005 0.15 2.0 5.0 512 B1_rho0.15_l02_lmax5_steps512
  run_ada 0.01005 0.1  2.0 5.0 512 B2_rho0.1_l02_lmax5_steps512
  run_ada 0.01005 0.2  2.0 5.0 512 B3_rho0.2_l02_lmax5_steps512
  echo "[$(date)] ROUND2 COMPLETE" | tee -a "$SUMMARY"

elif [ "$QUEUE" = "round3" ]; then
  # steps=512 refinement. Best so far: rho=0.15,l0=2,lmax=5 -> 5.72% (beats const 6.18%).
  # QED-recovery branch (gentler guidance, keep viol<6.18%, lift qed toward 0.479):
  run_ada 0.01005 0.15 2.0 4.0  512 D1_rho0.15_l02_lmax4_steps512
  run_ada 0.01005 0.15 1.0 5.0  512 D2_rho0.15_l01_lmax5_steps512
  run_ada 0.01005 0.12 2.0 5.0  512 D3_rho0.12_l02_lmax5_steps512
  run_ada 0.01005 0.15 2.0 4.5  512 D4_rho0.15_l02_lmax4.5_steps512
  # Viol-minimization branch (finer 512 grid may tolerate higher cap / faster ramp):
  run_ada 0.01005 0.15 2.0 6.0  512 D5_rho0.15_l02_lmax6_steps512
  run_ada 0.01005 0.18 2.0 5.0  512 D6_rho0.18_l02_lmax5_steps512
  run_ada 0.01005 0.2  2.0 4.0  512 D7_rho0.2_l02_lmax4_steps512
  echo "[$(date)] ROUND3 COMPLETE" | tee -a "$SUMMARY"
fi
echo "==================== QUEUE $QUEUE DONE @ $(date) ====================" | tee -a "$SUMMARY"
