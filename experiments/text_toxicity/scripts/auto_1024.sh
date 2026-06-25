#!/bin/bash
# ============================================================================
# Autonomous steps=1024 explorer (toxic prefix, N=1000, τ=0.5). Runs unattended
# in tmux for ~6h. Plan:
#   1. wait for the in-flight CBG γ=400 steps=1024 run to finish (no double-book),
#   2. fill the CBG curve: γ=150,250,300,350,
#   3. auto-pick adaptive-dual configs around the measured 1024 sweet spot γ*
#      (emit_adual_1024.py) and run them,
#   4. after every run, refresh the fluency-aware leaderboard (viol + collapse%).
# Stops launching new runs past DEADLINE (epoch, $1). Records per-run wall-clock.
# Launch: tmux new-session -d -s auto1024 \
#   'bash experiments/text_toxicity/scripts/auto_1024.sh <deadline_epoch>'
# ============================================================================
cd /local/scratch/zhiheng/guidance
LOGDIR=experiments/text_toxicity/logs; mkdir -p "$LOGDIR"
LOG=$LOGDIR/auto_1024_driver.log; STATUS=$LOGDIR/auto_1024_STATUS.txt
PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
SUMM="$PY experiments/text_toxicity/analysis/summarize_steps1024.py"
export HF_HOME=/home/tan.1422/.cache/huggingface PYTHONPATH=/local/scratch/zhiheng/guidance TOKENIZERS_PARALLELISM=false
# /tmp is full (cursor-sandbox-cache); route all scratch temp off /tmp.
export TMPDIR=/local/scratch/zhiheng/tmp
export MPLCONFIGDIR=/local/scratch/zhiheng/tmp/mpl
export TORCHINDUCTOR_CACHE_DIR=/local/scratch/zhiheng/tmp/inductor
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR"
OUT=outputs/owt/mdlm_owt
DEADLINE=${1:-0}
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
status(){ echo "[$(date '+%F %T')] $*" > "$STATUS"; }
pick_gpu(){ nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | awk -F, '$1<4 && $2+0>30000{print $2,$1}' | sort -rn | head -1 | awk '{print $2}'; }

run() {  # run <kind> <name-for-skip-check> <args...>
  local kind=$1; shift
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then log "DEADLINE reached — skip $kind $*"; return 1; fi
  local f desc
  if [ "$kind" = cbg ]; then
    f=$OUT/tox_cbg_gamma$1_le0.50_toxpfx_n1000_steps1024_results.csv; desc="CBG γ=$1"
  else
    f=$OUT/tox_adual_C$1_rho$2_l0$3_lmax$4_le0.50_toxpfx_n1000_steps1024_results.csv
    desc="adual C=$1 ρ=$2 λ0=$3 λmax=$4"
  fi
  [ -f "$f" ] && { log "skip (done): $desc"; return 0; }
  local G; G=$(pick_gpu)
  local waited=0
  while [ -z "$G" ]; do
    [ "$(date +%s)" -ge "$DEADLINE" ] && { log "DEADLINE while waiting GPU — skip $desc"; return 1; }
    sleep 120; waited=$((waited+120)); G=$(pick_gpu)
  done
  local s=$(date +%s); status "RUNNING $desc on GPU $G"
  log "RUN $desc on GPU $G"
  if [ "$kind" = cbg ]; then
    PROMPT_SELECT=top TAU=0.50 CUDA_VISIBLE_DEVICES=$G bash experiments/text_toxicity/scripts/sample_tox_cbg.sh "$1" 1000 16 1024 >>"$LOG" 2>&1
  else
    PROMPT_SELECT=top TAU=0.50 CUDA_VISIBLE_DEVICES=$G bash experiments/text_toxicity/scripts/sample_tox_adaptive.sh "$1" "$2" "$3" "$4" 1000 16 1024 >>"$LOG" 2>&1
  fi
  log "DONE $desc | $(( $(date +%s)-s ))s"
  $SUMM >>"$LOG" 2>&1
}

log "=== auto_1024 started; deadline=$(date -d @$DEADLINE '+%F %T' 2>/dev/null) ==="
status "started"

# 1) wait for in-flight γ=400 (current run_in_background job)
log "waiting for in-flight CBG γ=400 steps=1024…"
until [ -f $OUT/tox_cbg_gamma400_le0.50_toxpfx_n1000_steps1024_results.csv ]; do
  sleep 120; [ "$(date +%s)" -ge "$DEADLINE" ] && { log "deadline before γ=400 done"; break; }
done
sleep 30

# 2) CBG fill-in
for g in 150 250 300 350; do run cbg "$g"; done
$SUMM >>"$LOG" 2>&1

# 3) adual exploration around measured γ*
$PY experiments/text_toxicity/analysis/emit_adual_1024.py > /tmp/_adual1024.txt 2>>"$LOG"
log "adual configs (from γ*): $(tr '\n' ' ' < /tmp/_adual1024.txt)"
while read -r C rho l0 lmax; do
  [ -n "$C" ] && run adual "$C" "$rho" "$l0" "$lmax"
done < /tmp/_adual1024.txt

$SUMM >>"$LOG" 2>&1
VERDICT=$(grep -aE "Lowest Viol|Fluency-clean best" experiments/text_toxicity/results/leaderboard_steps1024.md 2>/dev/null)
log "=== AUTO_1024 DONE ==="
status "DONE
$VERDICT"
