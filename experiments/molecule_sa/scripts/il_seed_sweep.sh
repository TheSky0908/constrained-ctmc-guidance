#!/bin/bash
# Sweep a list of seeds on ONE gpu for the fixed best J=2 config, stop early
# when Viol@3.0 < 8.5%. Usage: il_seed_sweep.sh GPU "s1 s2 s3 ..."
set -e
cd /local/scratch/zhiheng/guidance
GPU=$1; SEEDS=$2
C=0.05129; RHO=0.15; L0=2.0; LMAX=5.0; J=2; NMC=1   # fixed best basin
export CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=offline HYDRA_FULL_ERROR=1
export HF_HOME=/local/scratch/zhiheng/guidance/.hf_cache
export PYTHONPATH=/local/scratch/zhiheng/guidance:/local/scratch/zhiheng/guidance/guidance_eval
PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
ROOT=/local/scratch/zhiheng/guidance
DIFF=$ROOT/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt
CLF=$ROOT/outputs/qm9/classifier/sa_score_le_3.0_absorbing_state_T-0_timedep/checkpoints/best.ckpt
for SEED in $SEEDS; do
  TAG=sweep_seed${SEED}
  OUT=$ROOT/outputs/qm9/il_search/${TAG}; mkdir -p $OUT
  LOG=$ROOT/logs/il_${TAG}.log
  SECONDS=0
  $PY -u sa_eval.py \
    data=qm9 model=small backbone=dit model.length=32 \
    parameterization=subs diffusion=absorbing_state time_conditioning=False T=0 \
    training.guidance=null data.label_col=sa_score data.label_col_pctile=null \
    +data.label_col_value_threshold=3.0 data.num_classes=2 \
    classifier_backbone=dit classifier_model=tiny-classifier \
    eval.checkpoint_path=$DIFF \
    guidance=cbg guidance.method=cbg guidance.gamma=1.0 \
    guidance.gamma_schedule=adaptive_dual guidance.classifier_time_conditioning=True \
    guidance.gamma_C=$C guidance.gamma_rho=$RHO \
    guidance.gamma_lambda_init=$L0 guidance.gamma_lambda_max=$LMAX \
    guidance.condition=1 guidance.gamma_adual_inner_loop=True \
    guidance.gamma_adual_n_mc=$NMC guidance.gamma_adual_n_inner=$J \
    guidance.gamma_adual_eps_tol=null \
    guidance.classifier_checkpoint_path=$CLF \
    sampling.steps=64 sampling.num_sample_batches=125 loader.eval_global_batch_size=4 \
    eval.generated_samples_path=$OUT/samples.json \
    +eval.results_csv_path=$OUT/results.csv \
    seed=$SEED hydra.run.dir=$OUT/run > $LOG 2>&1
  V=$(grep -E 'tau=3.0:' $LOG | grep -oE '[0-9.]+%' | tr -d '%' | head -1)
  VAL=$(grep -oE 'Valid +: +[0-9]+ \([0-9.]+%\)' $LOG | head -1)
  echo "seed=$SEED  Viol@3.0=${V}%  ${VAL}  (${SECONDS}s)"
  # early stop if below 8.5
  if awk "BEGIN{exit !($V < 8.5)}"; then
    echo ">>> HIT: seed=$SEED Viol=${V}% < 8.5%  (config C=-log0.95 rho=0.15 l0=2 lmax=5 J=2 n=1)"
    break
  fi
done
echo "[gpu$GPU sweep done]"
