#!/bin/bash
# v6: sampling.steps=2048 × C∈{-log 0.99, 0} × ρ=0.1.
# Wait-then-launch on free GPU in {0,1,2,3}.
set -e
cd /local/scratch/zhiheng/guidance
mkdir -p logs

N_BATCHES=125
BS=4
TAU=3.0
TAG=n500
STEPS=2048
ME=$(whoami)

find_free_gpu() {
  while true; do
    for g in 0 1 2 3; do
      util_mem=$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
                            --format=csv,noheader,nounits -i $g 2>/dev/null | tr -d ' ')
      util=${util_mem%,*}
      mem=${util_mem##*,}
      pids=$(nvidia-smi --query-compute-apps=pid \
                        --format=csv,noheader,nounits -i $g 2>/dev/null)
      foreign=0
      for pid in $pids; do
        if [ -n "$pid" ]; then
          user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
          if [ -n "$user" ] && [ "$user" != "$ME" ]; then foreign=1; break; fi
        fi
      done
      if [ "$util" -lt 5 ] && [ "$mem" -lt 1000 ] && [ "$foreign" -eq 0 ]; then
        echo $g; return 0
      fi
    done
    echo "[$(date)] no free GPU in {0,1,2,3}; sleeping 60s" >&2
    sleep 60
  done
}

declare -a RUNS=(
  "0.01005 0.1 0.0 50.0"
  "0.0     0.1 0.0 50.0"
)

for r in "${RUNS[@]}"; do
  set -- $r
  C=$1; RHO=$2; L0=$3; LMAX=$4
  echo "[$(date)] === waiting for free GPU for C=$C ρ=$RHO steps=$STEPS ==="
  GPU=$(find_free_gpu)
  echo "[$(date)] === claimed GPU $GPU; launching C=$C ρ=$RHO steps=$STEPS ==="
  LOG=logs/sa_dcbg_adual_${TAG}_C${C}_rho${RHO}_steps${STEPS}_trainTau${TAU}.log
  CUDA_VISIBLE_DEVICES=$GPU bash sample_sa_dcbg_adaptive.sh \
      $C $RHO $L0 $LMAX $N_BATCHES $BS $TAU $TAG $STEPS > $LOG 2>&1
  grep -E "adaptive_dual\] step|Valid               :|Viol|Novel|wrote adaptive" $LOG | head -25
done

echo "[$(date)] all v6 runs done"
