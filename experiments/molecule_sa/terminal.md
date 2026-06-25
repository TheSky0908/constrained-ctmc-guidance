
export PYTHONPATH="$PWD"


## train unconditional mdlm sampler
run3 -u -m main \
  diffusion=absorbing_state \
  parameterization=subs \
  T=0 \
  time_conditioning=False \
  zero_recon_loss=False \
  data=qm9 \
  data.cache_dir="$PWD/.data_cache" \
  data.label_col=null \
  data.label_col_pctile=null \
  data.num_classes=null \
  eval.generate_samples=False \
  loader.global_batch_size=256 \
  loader.eval_global_batch_size=512 \
  backbone=dit \
  model=small \
  model.length=32 \
  optim.lr=3e-4 \
  training.guidance=null \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1000 \
  training.compute_loss_on_pad_tokens=True \
  training.use_simple_ce_loss=False \
  trainer.devices=1 \
  trainer.max_steps=25000 \
  trainer.val_check_interval=1.0 \
  wandb.name=qm9_mdlm_no-guidance \
  hydra.run.dir="$PWD/outputs/qm9/mdlm_no-guidance" \
  loader.num_workers=0 \
  loader.persistent_workers=False \
  

## unconditional sampler (regular)
```python
run4 -u guidance_eval/qm9_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="$PWD/outputs/qm9/mdlm_no-guidance" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  seed=1 \
  mode=qm9_eval \
  eval.checkpoint_path="$PWD/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt" \
  data=qm9 \
  data.cache_dir="$PWD/.data_cache" \
  data.label_col=qed \
  data.label_col_pctile=90 \
  data.num_classes=2 \
  model=small \
  backbone=dit \
  model.length=32 \
  training.guidance=null \
  parameterization=subs \
  diffusion=absorbing_state \
  time_conditioning=False \
  T=0 \
  sampling.num_sample_batches=25 \
  sampling.batch_size=40 \
  sampling.steps=32 \
  sampling.use_cache=True \
  +eval.results_csv_path="$PWD/outputs/qm9/mdlm_no-guidance/eval_results.csv" \
  eval.generated_samples_path="$PWD/outputs/qm9/mdlm_no-guidance/samples.json"
```


## unconditional sampler (fhs)

run4 -u fhs.py \
  hydra.output_subdir=null \
  hydra.run.dir="$PWD/outputs/qm9/mdlm_no-guidance" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  seed=1 \
  data=qm9 \
  data.cache_dir="$PWD/.data_cache" \
  data.label_col=qed \
  data.label_col_pctile=90 \
  data.num_classes=2 \
  model=small \
  backbone=dit \
  model.length=32 \
  training.guidance=null \
  parameterization=subs \
  diffusion=absorbing_state \
  time_conditioning=False \
  T=0 \
  eval.checkpoint_path="$PWD/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt" \
  sampling.num_sample_batches=25 \
  loader.eval_global_batch_size=40 \
  eval.generated_samples_path="$PWD/outputs/qm9/mdlm_no-guidance/fhs_samples.json" \
  +eval.results_csv_path="$PWD/outputs/qm9/mdlm_no-guidance/fhs_eval_results.csv"


## unconditional sampler (euler)

run4 -u euler.py \
  data=qm9 \
  training.guidance=null \
  model.length=32 \
  eval.checkpoint_path="$PWD/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt" \
  sampling.steps=32 \
  sampling.num_sample_batches=25 \
  loader.eval_global_batch_size=40 \
  eval.generated_samples_path="$PWD/outputs/qm9/mdlm_no-guidance/euler_samples.json" \
  +eval.results_csv_path="$PWD/outputs/qm9/mdlm_no-guidance/euler_eval_results.csv"



## train discriminator

run4 -u discriminator_train.py \
  hydra.output_subdir=null \
  hydra.run.dir="$PWD/outputs/qm9/mdlm_no-guidance" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  seed=1 \
  data=qm9 \
  data.cache_dir="$PWD/.data_cache" \
  model=small \
  backbone=dit \
  model.length=32 \
  training.guidance=null \
  parameterization=subs \
  diffusion=absorbing_state \
  time_conditioning=False \
  T=0 \
  eval.checkpoint_path="$PWD/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt" \
  discriminator.save_dir="$PWD/outputs/qm9/discriminator" \
  discriminator.num_steps=50000 \
  discriminator.batch_size=16


## constrained fhs sampler (hard novelty constraint)

run4 -u constrained_fhs.py \
  hydra.output_subdir=null \
  hydra.run.dir="$PWD/outputs/qm9/mdlm_no-guidance" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  seed=1 \
  data=qm9 \
  data.cache_dir="$PWD/.data_cache" \
  data.label_col=qed \
  data.label_col_pctile=90 \
  data.num_classes=2 \
  model=small \
  backbone=dit \
  model.length=32 \
  training.guidance=null \
  parameterization=subs \
  diffusion=absorbing_state \
  time_conditioning=False \
  T=0 \
  eval.checkpoint_path="$PWD/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt" \
  discriminator.checkpoint_path="$PWD/outputs/qm9/discriminator/h_phi_final.pt" \
  discriminator.epsilon=0.10 \
  sampling.num_sample_batches=25 \
  loader.eval_global_batch_size=40 \
  eval.generated_samples_path="$PWD/outputs/qm9/mdlm_no-guidance/constrained_fhs_samples.json" \
  +eval.results_csv_path="$PWD/outputs/qm9/mdlm_no-guidance/constrained_fhs_results.csv"


## constrained ddpm sampler (hard novelty constraint)

run4 -u constrained_ddpm.py \
  hydra.output_subdir=null \
  hydra.run.dir="$PWD/outputs/qm9/mdlm_no-guidance" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  seed=1 \
  data=qm9 \
  data.cache_dir="$PWD/.data_cache" \
  data.label_col=qed \
  data.label_col_pctile=90 \
  data.num_classes=2 \
  model=small \
  backbone=dit \
  model.length=32 \
  training.guidance=null \
  parameterization=subs \
  diffusion=absorbing_state \
  time_conditioning=False \
  T=0 \
  eval.checkpoint_path="$PWD/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt" \
  discriminator.checkpoint_path="$PWD/outputs/qm9/discriminator/h_phi_final.pt" \
  discriminator.epsilon=0.10 \
  sampling.steps=32 \
  sampling.num_sample_batches=25 \
  loader.eval_global_batch_size=40 \
  eval.generated_samples_path="$PWD/outputs/qm9/mdlm_no-guidance/constrained_ddpm_samples.json" \
  +eval.results_csv_path="$PWD/outputs/qm9/mdlm_no-guidance/constrained_ddpm_results.csv"


## constrained euler sampler (hard novelty constraint)

run4 -u constrained_euler.py \
  hydra.output_subdir=null \
  hydra.run.dir="$PWD/outputs/qm9/mdlm_no-guidance" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  seed=1 \
  data=qm9 \
  data.cache_dir="$PWD/.data_cache" \
  data.label_col=qed \
  data.label_col_pctile=90 \
  data.num_classes=2 \
  model=small \
  backbone=dit \
  model.length=32 \
  training.guidance=null \
  parameterization=subs \
  diffusion=absorbing_state \
  time_conditioning=False \
  T=0 \
  eval.checkpoint_path="$PWD/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt" \
  discriminator.checkpoint_path="$PWD/outputs/qm9/discriminator/h_phi_final.pt" \
  discriminator.epsilon=0.10 \
  sampling.steps=32 \
  sampling.num_sample_batches=25 \
  loader.eval_global_batch_size=40 \
  eval.generated_samples_path="$PWD/outputs/qm9/mdlm_no-guidance/constrained_euler_samples.json" \
  +eval.results_csv_path="$PWD/outputs/qm9/mdlm_no-guidance/constrained_euler_results.csv"
