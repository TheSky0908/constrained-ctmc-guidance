#!/bin/bash
# Toxicity mitigation: mdlm-owt + D-CBG (constant gamma) guiding toward the
# NON-TOXIC class (condition=0). Generates prefix-conditioned continuations of
# RealToxicityPrompts prefixes and reports violation@{0.25,0.5,0.75} + PPL.
#
# Uses the first-order (use_approx=True) D-CBG path — REQUIRED at GPT-2's ~50k
# vocab (the exact-expansion path is O(B*L*V) and infeasible here).
#
# Usage:
#   bash sample_tox_cbg.sh [GAMMA] [N_PROMPTS] [BATCH] [STEPS]
# Defaults: GAMMA=3, N=100, BATCH=16, STEPS=128

set -eo pipefail
cd /local/scratch/zhiheng/guidance

GAMMA=${1:-3}
NUM_PROMPTS=${2:-100}
BATCH=${3:-16}
STEPS=${4:-128}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/tan.1422/.cache/huggingface
export PYTHONPATH=/local/scratch/zhiheng/guidance

PY=/home/tan.1422/zhiheng/miniconda3/envs/mdlm/bin/python
# TAU=<0.25|0.50|0.75> is REQUIRED: guidance uses a per-tau classifier
# (class 1 = score<=tau = "acceptable", so condition=1). The old binary
# main classifier has been dropped from this pipeline.
TAU=${TAU:?"set TAU=<0.25|0.50|0.75> (per-tau classifier; main 0/1 classifier was dropped)"}
CLF=/local/scratch/zhiheng/guidance/outputs/owt/classifier/toxicity_le_${TAU}_absorbing_state_T-0/checkpoints/best.ckpt
COND=1
TAG=_le${TAU}
# PROMPT_SELECT=top picks the most-toxic RTP prefixes (raises the unguided floor);
# PROMPT_TOX_MIN sets the prefix toxicity cutoff. Defaults reproduce prior runs.
PROMPT_SELECT=${PROMPT_SELECT:-random}
PROMPT_TOX_MIN=${PROMPT_TOX_MIN:-0.5}
PFXTAG=""; [ "$PROMPT_SELECT" = top ] && PFXTAG=_toxpfx
SURR=/local/scratch/zhiheng/guidance/outputs/owt/toxic_surrogate
OUT_DIR=/local/scratch/zhiheng/guidance/outputs/owt/mdlm_owt
LOGDIR=/local/scratch/zhiheng/guidance/experiments/text_toxicity/logs
mkdir -p "$OUT_DIR" "$LOGDIR"

RUN_NAME=tox_cbg_gamma${GAMMA}${TAG}${PFXTAG}_n${NUM_PROMPTS}_steps${STEPS}
SAMPLES_JSON=$OUT_DIR/${RUN_NAME}_samples.json
CSV=$OUT_DIR/${RUN_NAME}_results.csv
LOG=$LOGDIR/${RUN_NAME}.log

if [ ! -f "$CLF" ]; then echo "MISSING classifier: $CLF" >&2; exit 1; fi

exec > >(tee -a "$LOG") 2>&1   # self-log to experiments/text_toxicity/logs/
echo "[$(date)] mdlm-owt + D-CBG (gamma=$GAMMA) on GPU=$CUDA_VISIBLE_DEVICES, N=$NUM_PROMPTS"

$PY -u tox_eval.py \
  data=realtoxicityprompts \
  backbone=hf_mdlm \
  model=hf \
  model.pretrained_model_name_or_path=kuleshov-group/mdlm-owt \
  model.length=128 \
  parameterization=subs \
  diffusion=absorbing_state \
  time_conditioning=False \
  T=0 \
  data.num_classes=2 \
  classifier_backbone=dit \
  classifier_model=tiny-classifier \
  guidance=cbg \
  guidance.method=cbg \
  guidance.gamma=$GAMMA \
  guidance.condition=$COND \
  guidance.use_approx=True \
  guidance.classifier_checkpoint_path=$CLF \
  sampling.steps=$STEPS \
  loader.eval_global_batch_size=$BATCH \
  eval.disable_ema=True \
  eval.generated_samples_path=$SAMPLES_JSON \
  +eval.results_csv_path=$CSV \
  +tox.num_prompts=$NUM_PROMPTS \
  +tox.prompt_toxicity_min=$PROMPT_TOX_MIN \
  +tox.prompt_select=$PROMPT_SELECT \
  +tox.max_prefix_len=32 \
  +tox.surrogate_dir=$SURR \
  seed=1 \
  hydra.run.dir=$OUT_DIR/eval_${RUN_NAME}

echo "[$(date)] done. csv=$CSV"
