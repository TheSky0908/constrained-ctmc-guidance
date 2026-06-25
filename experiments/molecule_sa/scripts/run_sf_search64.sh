#!/bin/bash
# Reusable launcher for the adaptive-dual SAMPLE_FIRST (Algm 1) search at STEPS=64.
# Setting: adual sf + time-dep classifier, steps=64, N=500 (125 batches × 4).
# Usage: run_sf_search64.sh GPU C RHO L0 LMAX SEED TAG
set -e
cd /local/scratch/zhiheng/guidance
GPU=$1; C=$2; RHO=$3; L0=$4; LMAX=$5; SEED=$6; TAG=$7
export CUDA_VISIBLE_DEVICES=$GPU
export WANDB_MODE=offline HYDRA_FULL_ERROR=1
export HF_HOME=/local/scratch/zhiheng/guidance/.hf_cache
export PYTHONPATH=/local/scratch/zhiheng/guidance:/local/scratch/zhiheng/guidance/guidance_eval
PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
ROOT=/local/scratch/zhiheng/guidance
DIFF=$ROOT/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt
CLF=$ROOT/outputs/qm9/classifier/sa_score_le_3.0_absorbing_state_T-0_timedep/checkpoints/best.ckpt
OUT=$ROOT/outputs/qm9/sf_search64/${TAG}; mkdir -p $OUT $ROOT/logs
LOG=$ROOT/logs/sf_search64_${TAG}.log
echo "[$TAG] GPU=$GPU C=$C rho=$RHO l0=$L0 lmax=$LMAX seed=$SEED (sample_first, steps=64)"
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
  guidance.gamma_adual_update_order=sample_first \
  guidance.gamma_adual_inner_loop=False \
  guidance.classifier_time_conditioning=True \
  guidance.gamma_C=$C guidance.gamma_rho=$RHO \
  guidance.gamma_lambda_init=$L0 guidance.gamma_lambda_max=$LMAX \
  guidance.condition=1 \
  guidance.classifier_checkpoint_path=$CLF \
  sampling.steps=64 sampling.num_sample_batches=125 loader.eval_global_batch_size=4 \
  eval.generated_samples_path=$OUT/samples.json \
  +eval.results_csv_path=$OUT/results.csv \
  seed=$SEED hydra.run.dir=$OUT/run > $LOG 2>&1
echo "[$TAG] EXIT=$? elapsed=$((SECONDS/60))m$((SECONDS%60))s"
echo "[$TAG] $(grep -E 'Valid +:' $LOG | head -1 | tr -s ' ')"
echo "[$TAG] $(grep -E 'tau=3.0:' $LOG | tr -s ' ')"
