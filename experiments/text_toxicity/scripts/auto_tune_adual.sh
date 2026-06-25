#!/bin/bash
# ============================================================================
# Autonomous adaptive-dual tuning driver (tau=0.5, steps=128, N=500 — all FIXED).
# Goal: find an adaptive-dual (C, rho, lambda0, lambda_max) that beats CBG gamma=3
# on Viol@0.50, with the best PPL among such configs.
#
# Fully self-contained: runs every config serially on one free GPU among 0-3,
# picks the freest card before each run, retries once on a different card on
# failure (OOM-safe), updates a leaderboard after EVERY run, and stops launching
# new runs once the time budget is exhausted (then writes a final summary).
#
# Survives SSH disconnect / laptop sleep because it runs detached in tmux. Needs
# NO further interaction. Launch:
#   tmux new-session -d -s tox_autotune \
#     'bash experiments/text_toxicity/scripts/auto_tune_adual.sh'
# ============================================================================

cd /local/scratch/zhiheng/guidance

LOGDIR=experiments/text_toxicity/logs
mkdir -p "$LOGDIR"
DRIVERLOG=$LOGDIR/auto_tune_driver.log
STATUS=$LOGDIR/auto_tune_STATUS.txt
PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
SUMMARIZE="$PY experiments/text_toxicity/analysis/summarize_tox_tuning.py"

export HF_HOME=/home/tan.1422/.cache/huggingface
export PYTHONPATH=/local/scratch/zhiheng/guidance
export TOKENIZERS_PARALLELISM=false

N=500; BATCH=16; STEPS=128; TAU=0.50
MEM_THRESH=30000          # MiB free required on a GPU before we use it
DEADLINE_SECS=$((6*3600 + 1800))   # stop launching new runs after 6.5h
GPU_WAIT_MAX=2400         # wait up to 40 min for a free GPU before skipping

log() { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVERLOG"; }
status() { echo "[$(date '+%F %T')] $*" > "$STATUS"; }

# Freest GPU among 0-3 with free mem > MEM_THRESH; echoes index or empty.
pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F, -v th="$MEM_THRESH" '{gsub(/ /,"",$1);gsub(/ /,"",$2)} $1<4 && $2+0>th {print $2, $1}' \
    | sort -rn | head -1 | awk '{print $2}'
}

wait_for_gpu() {
  local waited=0 g
  while :; do
    g=$(pick_gpu)
    if [ -n "$g" ]; then echo "$g"; return 0; fi
    if [ "$waited" -ge "$GPU_WAIT_MAX" ]; then echo ""; return 1; fi
    sleep 30; waited=$((waited+30))
  done
}

# run_one <kind> <args...>
#   cbg:   run_one cbg <GAMMA>
#   adual: run_one adual <C> <RHO> <L0> <LMAX>
run_one() {
  local kind=$1; shift
  local desc gpu rc
  if [ "$kind" = cbg ]; then
    desc="cbg gamma=$1"
    csv=outputs/owt/mdlm_owt/tox_cbg_gamma$1_le0.50_n${N}_steps${STEPS}_results.csv
  else
    desc="adual C=$1 rho=$2 l0=$3 lmax=$4"
    csv=outputs/owt/mdlm_owt/tox_adual_C$1_rho$2_l0$3_lmax$4_le0.50_n${N}_steps${STEPS}_results.csv
  fi
  if [ -f "$csv" ]; then log "SKIP (already done): $desc"; return 0; fi

  local attempt
  for attempt in 1 2; do
    gpu=$(wait_for_gpu)
    if [ -z "$gpu" ]; then log "NO free GPU for [$desc] after ${GPU_WAIT_MAX}s — skipping"; return 1; fi
    log "RUN [$desc] on GPU $gpu (attempt $attempt)"
    status "RUNNING: $desc on GPU $gpu (run #$RUN_IDX/$TOTAL)"
    if [ "$kind" = cbg ]; then
      TAU=$TAU CUDA_VISIBLE_DEVICES=$gpu bash experiments/text_toxicity/scripts/sample_tox_cbg.sh \
        "$1" "$N" "$BATCH" "$STEPS" >>"$DRIVERLOG" 2>&1
    else
      TAU=$TAU CUDA_VISIBLE_DEVICES=$gpu bash experiments/text_toxicity/scripts/sample_tox_adaptive.sh \
        "$1" "$2" "$3" "$4" "$N" "$BATCH" "$STEPS" >>"$DRIVERLOG" 2>&1
    fi
    rc=$?
    if [ "$rc" -eq 0 ] && [ -f "$csv" ]; then
      log "DONE [$desc]"; $SUMMARIZE >>"$DRIVERLOG" 2>&1; return 0
    fi
    log "FAIL [$desc] rc=$rc (attempt $attempt) — will retry on another GPU"
    sleep 20
  done
  log "GIVE UP [$desc] after 2 attempts"; return 1
}

# ---- Config list, highest-value first ---------------------------------------
# CBG baselines (gamma=3 first = the target, then the rest to locate the sweet spot)
CBG_GAMMAS=(3 2 1 4 5)
# Adaptive-dual: "C RHO L0 LMAX". Hypotheses:
#   - tight C=0.01005 (=-log0.99, unreachable): dual rides cap -> cap near sweet spot.
#   - cap lambda_max at/below the constant-g sweet spot to kill overshoot.
#   - warm-start lambda0 to remove the early under-guided window.
#   - small rho (0.1) for a smooth update on the coarse 128-step grid.
#   - reachable C (0.105/0.223/0.511 = -log 0.9/0.8/0.6) so the dual actually
#     relaxes on satisfied samples (true per-sample adaptivity = the real win).
ADUAL_CFGS=(
  "0.01005 0.1 2 3"
  "0.01005 0.1 3 3"
  "0.01005 0.1 2 2"
  "0.01005 0.2 2 3"
  "0.223 0.1 2 3"
  "0.105 0.1 2 3"
  "0.01005 0.1 2 4"
  "0.223 0.2 2 4"
  "0.105 0.2 2 4"
  "0.01005 0.2 3 3"
  "0.511 0.1 2 3"
  "0.01005 0.1 0 3"
  "0.01005 0.2 0 3"
  "0.223 0.1 2 5"
  "0.105 0.1 3 4"
  "0.01005 0.1 3 4"
  "0.511 0.2 2 4"
  "0.223 0.1 3 3"
  "0.105 0.1 2 2"
  "0.01005 0.1 2 5"
)

TOTAL=$(( ${#CBG_GAMMAS[@]} + ${#ADUAL_CFGS[@]} ))
RUN_IDX=0
SECONDS=0

log "=== auto-tune started: $TOTAL configs, N=$N, deadline ${DEADLINE_SECS}s ==="
status "STARTED: $TOTAL configs queued"
$SUMMARIZE >>"$DRIVERLOG" 2>&1   # baseline leaderboard (may be empty)

for g in "${CBG_GAMMAS[@]}"; do
  RUN_IDX=$((RUN_IDX+1))
  [ "$SECONDS" -ge "$DEADLINE_SECS" ] && { log "DEADLINE hit before run #$RUN_IDX — stopping"; break; }
  run_one cbg "$g"
done

for cfg in "${ADUAL_CFGS[@]}"; do
  RUN_IDX=$((RUN_IDX+1))
  [ "$SECONDS" -ge "$DEADLINE_SECS" ] && { log "DEADLINE hit before run #$RUN_IDX — stopping"; break; }
  # shellcheck disable=SC2086
  run_one adual $cfg
done

log "=== all queued runs attempted; writing final summary ==="
$SUMMARIZE >>"$DRIVERLOG" 2>&1
VERDICT=$(grep -A6 '## Verdict' experiments/text_toxicity/results/tuning_leaderboard.md 2>/dev/null | tail -n +2)
status "DONE after ${SECONDS}s. See results/tuning_leaderboard.md
$VERDICT"
log "=== AUTO-TUNE DONE (${SECONDS}s) ==="
log "$VERDICT"
