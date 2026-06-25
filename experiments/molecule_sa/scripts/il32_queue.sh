#!/bin/bash
# Robust 2-concurrent queue for il32 tuning on GPU7. Args: list of "C RHO L0 LMAX J N SEED TAG" rows in a file.
# Usage: il32_queue.sh ROWFILE MAXPAR
ROWFILE=$1; MAXPAR=${2:-2}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SC=/local/scratch/zhiheng/guidance/experiments/molecule_sa/scripts/run_il_search_steps.sh
cd /local/scratch/zhiheng/guidance/experiments/molecule_sa
while IFS= read -r row; do
  [ -z "$row" ] && continue
  set -- $row; C=$1; RHO=$2; L0=$3; LMAX=$4; J=$5; N=$6; SEED=$7; TAG=$8
  # throttle to MAXPAR
  while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do sleep 10; done
  echo "[queue] launch $TAG"
  nohup bash $SC 7 $C $RHO $L0 $LMAX $J $N $SEED $TAG 32 125 > logs/${TAG}.out 2>&1 &
  sleep 30  # stagger starts to avoid simultaneous memory peaks
done < "$ROWFILE"
wait
echo "[queue] ALL DONE"
