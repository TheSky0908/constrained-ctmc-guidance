#!/bin/bash
# Wait for the running steps=512 adual sweep (tmux: novelty_adual_s512) to finish,
# then run CBG gamma 0-10 @ steps=1024 on whichever of GPU 0-3 is freest.
# Respects one-card-at-a-time: never overlaps with the 512 job. Persisted in tmux.
set -uo pipefail
cd /local/scratch/zhiheng/guidance
LOG=experiments/molecule_novelty/logs/sweep_gamma_steps1024_progress.log

echo "[$(date '+%F %T')] chain_1024: waiting for novelty_adual_s512 to finish..."
while tmux has-session -t novelty_adual_s512 2>/dev/null; do sleep 120; done
echo "[$(date '+%F %T')] 512 adual finished. selecting GPU..."

# pick freest GPU among 0-3 (lowest used memory); wait until one is reasonably free
GPU=""
for attempt in $(seq 1 60); do
  read IDX MEM <<< "$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | head -4 | sort -t, -k2 -n | head -1 | tr ',' ' ')"
  if [ "${MEM:-999999}" -lt 20000 ]; then GPU=$IDX; break; fi
  echo "[$(date '+%F %T')] no free GPU (best=GPU$IDX ${MEM}MiB); retry in 120s..."
  sleep 120
done
GPU=${GPU:-3}
echo "[$(date '+%F %T')] launching gamma 0-10 @ steps=1024 on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU bash experiments/molecule_novelty/scripts/sweep_gamma.sh \
  "0 1 2 3 4 5 6 7 8 9 10" 1024 2>&1 | tee "$LOG"
echo "[$(date '+%F %T')] chain_1024 done."
