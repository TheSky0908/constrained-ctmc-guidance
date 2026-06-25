#!/bin/bash
# Reusable launcher for the inner-loop (Algm 2) hyperparameter search.
# Setting: adual IL + time-dep classifier, steps=64, N=500 (125 batches × 4).
# Usage: run_il_search.sh GPU C RHO L0 LMAX J N SEED TAG
set -e
cd /local/scratch/zhiheng/guidance
GPU=$1; C=$2; RHO=$3; L0=$4; LMAX=$5; J=$6; NMC=$7; SEED=$8; TAG=$9
export CUDA_VISIBLE_DEVICES=$GPU
export WANDB_MODE=offline HYDRA_FULL_ERROR=1
export HF_HOME=/local/scratch/zhiheng/guidance/.hf_cache
export PYTHONPATH=/local/scratch/zhiheng/guidance:/local/scratch/zhiheng/guidance/guidance_eval
PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
ROOT=/local/scratch/zhiheng/guidance
DIFF=$ROOT/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt
CLF=$ROOT/outputs/qm9/classifier/sa_score_le_3.0_absorbing_state_T-0_timedep/checkpoints/best.ckpt
OUT=$ROOT/outputs/qm9/il_search/${TAG}
mkdir -p $OUT
LOG=$ROOT/logs/il_search_${TAG}.log
echo "[$TAG] GPU=$GPU C=$C rho=$RHO l0=$L0 lmax=$LMAX J=$J n=$NMC seed=$SEED"
SECONDS=0
$PY -u sa_eval.py \
  data=qm9 model=small backbone=dit model.length=32 \
  parameterization=subs diffusion=absorbing_state time_conditioning=False T=0 \
  training.guidance=null data.label_col=sa_score data.label_col_pctile=null \
  +data.label_col_value_threshold=3.0 data.num_classes=2 \
  classifier_backbone=dit classifier_model=tiny-classifier \
  eval.checkpoint_path=$DIFF \
  guidance=cbg guidance.method=cbg guidance.gamma=1.0 \
  guidance.gamma_schedule=adaptive_dual \
  guidance.classifier_time_conditioning=True \
  guidance.gamma_C=$C guidance.gamma_rho=$RHO \
  guidance.gamma_lambda_init=$L0 guidance.gamma_lambda_max=$LMAX \
  guidance.condition=1 \
  guidance.gamma_adual_inner_loop=True \
  guidance.gamma_adual_n_mc=$NMC guidance.gamma_adual_n_inner=$J \
  guidance.gamma_adual_eps_tol=null \
  guidance.classifier_checkpoint_path=$CLF \
  sampling.steps=64 sampling.num_sample_batches=125 \
  loader.eval_global_batch_size=4 \
  eval.generated_samples_path=$OUT/samples.json \
  +eval.results_csv_path=$OUT/results.csv \
  seed=$SEED hydra.run.dir=$OUT/run > $LOG 2>&1
EC=$?
echo "[$TAG] EXIT=$EC elapsed=$((SECONDS/60))m$((SECONDS%60))s"
echo "[$TAG] $(grep -oE 'Valid +: +[0-9]+ \([0-9.]+%\)' $LOG | head -1)"
echo "[$TAG] $(grep -E 'tau=3.0:' $LOG | head -1)"
