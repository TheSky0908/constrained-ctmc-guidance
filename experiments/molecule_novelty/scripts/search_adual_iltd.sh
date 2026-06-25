#!/bin/bash
# Iterative parameter search driver for adaptive-dual INNER-LOOP (Algm 2) + td clf
# on the NOVELTY task. Runs a list of (C,rho,l0,lmax,J,n,seed[,steps]) combos
# SEQUENTIALLY on one GPU, times each, and appends a row to a shared master CSV.
# Idempotent: skips a combo whose per-run results CSV already exists.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash search_adual_iltd.sh "C:RHO:L0:LMAX:J:N:SEED[:STEPS]" ["..." ...]
#   (steps default 64, N=500. target p(novel)=e^{-C}. J=n_inner>=2, n=n_mc.)
#
# Inner-loop dual (Algorithm 2): at each step freeze x_t, solve lambda* by
#   projected grad ascent with an n-sample MC estimate over up to J iters,
#   then advance x_{t+1} with lambda*.

set -uo pipefail
cd /local/scratch/zhiheng/guidance

DEFAULT_STEPS=64
NUM_BATCHES=125
BATCH_SIZE=4

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export HF_HOME=/local/scratch/zhiheng/guidance/.hf_cache
export PYTHONPATH=/local/scratch/zhiheng/guidance:/local/scratch/zhiheng/guidance/guidance_eval

PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
DIFF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt
CLF_CKPT=/local/scratch/zhiheng/guidance/outputs/qm9/classifier/novelty_rollout_timedep/checkpoints/best.ckpt
OUT_DIR=/local/scratch/zhiheng/guidance/experiments/molecule_novelty/results
SEARCH_DIR=$OUT_DIR/adual_iltd_search
EVAL=experiments/molecule_novelty/novelty_eval.py
LOGDIR=/local/scratch/zhiheng/guidance/experiments/molecule_novelty/logs
mkdir -p "$SEARCH_DIR" "$LOGDIR"
MASTER_HDR="C,rho,l0,lmax,J,n,seed,steps,seconds,valid,unique,valid_novel,viol,novel_rate,qed,run_name"

for p in "$DIFF_CKPT" "$CLF_CKPT"; do
  [ -f "$p" ] || { echo "[search_adual_iltd] MISSING CKPT: $p" >&2; exit 1; }
done

for combo in "$@"; do
  IFS=':' read -r C RHO L0 LMAX J NMC SEED STEPS <<< "$combo"
  SEED=${SEED:-1}
  SAMPLING_STEPS=${STEPS:-$DEFAULT_STEPS}
  MASTER=$OUT_DIR/adual_iltd_search_steps${SAMPLING_STEPS}.csv
  [ -f "$MASTER" ] || echo "$MASTER_HDR" > "$MASTER"
  RUN_NAME=adual_iltd_C${C}_rho${RHO}_l0${L0}_lmax${LMAX}_J${J}_n${NMC}_seed${SEED}_steps${SAMPLING_STEPS}
  CSV=$SEARCH_DIR/${RUN_NAME}_results.csv
  LOG=$LOGDIR/${RUN_NAME}.log
  if [ -f "$CSV" ]; then
    echo "[$(date '+%F %T')] [skip] $RUN_NAME (exists)"
    continue
  fi
  echo "[$(date '+%F %T')] === C=$C rho=$RHO l0=$L0 lmax=$LMAX J=$J n=$NMC seed=$SEED steps=$SAMPLING_STEPS GPU=$CUDA_VISIBLE_DEVICES (target p=e^-C) ==="
  T0=$(date +%s)
  $PY -u $EVAL \
    data=qm9 model=small backbone=dit model.length=32 \
    parameterization=subs diffusion=absorbing_state time_conditioning=False T=0 \
    training.guidance=null \
    data.label_col=novel data.label_col_pctile=null data.num_classes=2 \
    classifier_backbone=dit classifier_model=tiny-classifier \
    eval.checkpoint_path=$DIFF_CKPT \
    guidance=cbg guidance.method=cbg guidance.gamma=1.0 \
    guidance.gamma_schedule=adaptive_dual \
    guidance.classifier_time_conditioning=True \
    guidance.gamma_C=$C guidance.gamma_rho=$RHO \
    guidance.gamma_lambda_init=$L0 guidance.gamma_lambda_max=$LMAX \
    guidance.condition=1 \
    guidance.gamma_adual_inner_loop=True \
    guidance.gamma_adual_n_mc=$NMC guidance.gamma_adual_n_inner=$J \
    guidance.gamma_adual_eps_tol=null \
    guidance.classifier_checkpoint_path=$CLF_CKPT \
    sampling.steps=$SAMPLING_STEPS \
    sampling.num_sample_batches=$NUM_BATCHES \
    loader.eval_global_batch_size=$BATCH_SIZE \
    eval.generated_samples_path=$SEARCH_DIR/${RUN_NAME}_samples.json \
    +eval.results_csv_path=$CSV \
    seed=$SEED \
    hydra.run.dir=$SEARCH_DIR/eval_${RUN_NAME} \
    >> "$LOG" 2>&1
  RC=$?
  SEC=$(( $(date +%s) - T0 ))
  if [ $RC -eq 0 ] && [ -f "$CSV" ]; then
    L=$(tail -1 "$CSV")
    VALID=$(echo "$L" | cut -d, -f9); UNIQ=$(echo "$L" | cut -d, -f10)
    VN=$(echo "$L" | cut -d, -f11); QED=$(echo "$L" | cut -d, -f13)
    NR=$(echo "$L" | cut -d, -f14); VIOL=$(echo "$L" | cut -d, -f15)
    echo "$C,$RHO,$L0,$LMAX,$J,$NMC,$SEED,$SAMPLING_STEPS,$SEC,$VALID,$UNIQ,$VN,$VIOL,$NR,$QED,$RUN_NAME" >> "$MASTER"
    echo "[$(date '+%F %T')] [done] ${SEC}s valid=$VALID viol=$VIOL vn=$VN -> $MASTER"
  else
    echo "$C,$RHO,$L0,$LMAX,$J,$NMC,$SEED,$SAMPLING_STEPS,$SEC,FAIL,,,,,,$RUN_NAME" >> "$MASTER"
    echo "[$(date '+%F %T')] [FAIL] rc=$RC ${SEC}s — see $LOG"
  fi
done
echo "[$(date '+%F %T')] IL search batch done (GPU=$CUDA_VISIBLE_DEVICES)."
