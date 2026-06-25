#!/bin/bash
# Reusable launcher for the sf (sample_first) + time-dep classifier hyperparameter
# search at steps=128, N=500 (125 batches x 4). Mirrors run_il_search.sh but for the
# single-sample sample_first path (gamma_adual_inner_loop=False) and a tunable seed.
# Usage: run_sf_search.sh GPU C RHO L0 LMAX SEED TAG
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
OUT=$ROOT/outputs/qm9/sf_search/${TAG}
mkdir -p $OUT $ROOT/logs
LOG=$ROOT/logs/sf_search_${TAG}.log
echo "[$(date)] [$TAG] GPU=$GPU C=$C rho=$RHO l0=$L0 lmax=$LMAX seed=$SEED steps=128 N=500" | tee $LOG
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
  guidance.classifier_time_conditioning=True \
  guidance.gamma_C=$C guidance.gamma_rho=$RHO \
  guidance.gamma_lambda_init=$L0 guidance.gamma_lambda_max=$LMAX \
  guidance.condition=1 \
  guidance.classifier_checkpoint_path=$CLF \
  sampling.steps=128 sampling.num_sample_batches=125 \
  loader.eval_global_batch_size=4 \
  eval.generated_samples_path=$OUT/samples.json \
  +eval.results_csv_path=$OUT/results.csv \
  seed=$SEED hydra.run.dir=$OUT/run >> $LOG 2>&1
EC=$?
echo "[$(date)] [$TAG] EXIT=$EC elapsed=$((SECONDS/60))m$((SECONDS%60))s" | tee -a $LOG
echo "[$TAG] $(grep -E 'tau=3.0:' $LOG | head -1)  |  $(grep -E 'Valid +:' $LOG | head -1)" | tee -a $LOG
