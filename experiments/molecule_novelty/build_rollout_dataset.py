"""Build the on-disk `qm9_novel_rollout` dataset for the eq.(3) novelty
classifier (the "pretrained-model path").

The original `qm9_novel` classifier is trained on the FORWARD process: clean
SMILES x0 are noised on-the-fly via q(x_t|x_0). This builder instead realises
the pretrained-model path: it draws full *unconditional* trajectories from the
frozen base MDLM's own reverse process and labels EVERY intermediate state
x_{t_k} by the trajectory's own terminal outcome y = 1[x_{t_K} is novel]:

  L_model(phi) = - E_{traj ~ p_theta} E_k [ y log r_phi(x_{t_k}, t_k)
                                            + (1-y) log(1 - r_phi(x_{t_k}, t_k)) ]

Concretely:
  1. Roll out a batch of trajectories with the SAME absorbing-state DDPM sampler
     used at guidance time (`_ddpm_denoise`), snapshotting (x_t, t) at each step.
  2. Decode the terminal state x_0; keep only VALID molecules (mirrors
     build_novelty_dataset's label semantics); label novel=1 if the canonical
     SMILES is NOT in QM9-train, else 0.
  3. For each kept trajectory, emit `snapshots_per_traj` randomly-chosen step
     snapshots, each a row (input_ids=x_t, attention_mask=1s, t, novel=y).

Saved as a HF DatasetDict {train, validation} with columns
`input_ids` + `attention_mask` + `t` + `novel`, loaded by the
`qm9_novel_rollout` dataloader branch and consumed with
`training.classifier_rollout=True time_conditioning=True`.

Run (base MDLM is T=0 / time_conditioning=False; classifier is trained
time-DEPENDENT, so we roll out at a fine `sampling.steps`):
  python -u experiments/molecule_novelty/build_rollout_dataset.py \
    data=qm9 model=small backbone=dit model.length=32 \
    training.guidance=null parameterization=subs diffusion=absorbing_state \
    time_conditioning=False T=0 sampling.steps=128 \
    eval.checkpoint_path=outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt \
    +build.num_traj=8000 +build.snapshots_per_traj=8 +build.gen_batch_size=512 \
    +build.out_dir=.data_cache/qm9_novelty_rollout
"""
import itertools
import os

import datasets
import hydra
import lightning as L
import numpy as np
import omegaconf
import rdkit
import torch
from rdkit import Chem as rdChem
from tqdm.auto import tqdm

import dataloader
import diffusion

rdkit.rdBase.DisableLog('rdApp.error')

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd, replace=True)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count, replace=True)
omegaconf.OmegaConf.register_new_resolver('eval', eval, replace=True)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y, replace=True)
omegaconf.OmegaConf.register_new_resolver(
  'if_then_else',
  lambda condition, x, y: x if condition else y, replace=True)


@torch.no_grad()
def rollout_batch(model, batch_size, length, steps, eps):
  """One unconditional absorbing-state DDPM rollout, mirroring the no-guidance
  branch of diffusion._diffusion_sample.

  Returns:
    x0:     (B, L) long  — terminal (clean) states.
    snaps:  (steps, B, L) long  — the state x_t fed to the denoiser at each
            step i (i.e. exactly what the guidance classifier is evaluated on).
    t_vals: (steps,) float — the normalized step time t_i for each snapshot.
  """
  xt = model._sample_prior(batch_size, length).to(model.device)
  timesteps = torch.linspace(1, eps, steps + 1, device=model.device)
  dt = (1 - eps) / steps
  snaps = []
  t_vals = []
  for i in range(steps):
    t = timesteps[i]
    # Base novelty MDLM is T=0 (continuous), so no t-discretization is applied
    # (matches _diffusion_sample, whose `if self.T > 0` branch is skipped here).
    t_b = t * torch.ones(xt.shape[0], 1, device=model.device)
    sigma_t, _ = model.noise(t_b)
    sigma_s, _ = model.noise(t_b - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    move_chance_t = (1 - torch.exp(-sigma_t))[:, None, None]
    move_chance_s = (1 - torch.exp(-sigma_s))[:, None, None]
    # Snapshot the CURRENT state x_t at time t_i (what get_log_probs sees).
    snaps.append(xt.detach().cpu())
    t_vals.append(float(t))
    xs, _, _ = model._ddpm_denoise(
      xt=xt,
      time_conditioning=sigma_t,
      move_chance_t=move_chance_t,
      move_chance_s=move_chance_s,
      cache=None)
    xt = xs
  return xt.detach().cpu(), torch.stack(snaps, dim=0), t_vals


def _terminal_label(smiles, qm9_set):
  """Decode → validate → canonicalize one terminal SMILES.

  Returns (is_valid, novel). novel is 1 if valid & canonical ∉ QM9-train, else 0.
  """
  s = smiles.replace('<bos>', '').replace('<eos>', '').replace(
    '<pad>', '').strip()
  if not s:
    return False, 0
  try:
    mol = rdChem.MolFromSmiles(s)
  except Exception:
    mol = None
  if mol is None:
    return False, 0
  canon = rdChem.MolToSmiles(mol)
  if not canon:
    return False, 0
  return True, (0 if canon in qm9_set else 1)


@hydra.main(version_base=None, config_path='../../configs',
            config_name='config')
def main(config: omegaconf.DictConfig) -> None:
  L.seed_everything(config.seed)

  sel = lambda k, d: omegaconf.OmegaConf.select(config, k, default=d)
  num_traj = int(sel('build.num_traj', 8000))
  snaps_per_traj = int(sel('build.snapshots_per_traj', 8))
  gen_bs = int(sel('build.gen_batch_size', 512))
  max_batches = int(sel('build.max_gen_batches', 100000))
  val_frac = float(sel('build.val_frac', 0.05))
  out_dir = str(sel('build.out_dir', '.data_cache/qm9_novelty_rollout'))
  steps = int(config.sampling.steps)
  eps = float(sel('build.sampling_eps', 1e-5))
  seed = int(config.seed)
  rng = np.random.default_rng(seed)

  # ── QM9 novelty reference set ──────────────────────────────────────────────
  print('[build-rollout] loading QM9 train...')
  qm9 = datasets.load_dataset(
    'yairschiff/qm9', trust_remote_code=True, split='train')
  qm9_set = set(qm9['canonical_smiles'])
  print(f'[build-rollout] QM9 canonical SMILES: {len(qm9_set):,}')

  # ── frozen base MDLM ───────────────────────────────────────────────────────
  tokenizer = dataloader.get_tokenizer(config)
  model = diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path, tokenizer=tokenizer, config=config,
    logger=False)
  model.eval()
  if torch.cuda.is_available():
    model = model.to('cuda')
  for p in model.parameters():
    p.requires_grad_(False)
  if not getattr(config.eval, 'disable_ema', True) and model.ema is not None:
    model.ema.store(itertools.chain(
      model.backbone.parameters(), model.noise.parameters()))
    model.ema.copy_to(itertools.chain(
      model.backbone.parameters(), model.noise.parameters()))

  length = config.model.length

  # ── roll out + snapshot + label ────────────────────────────────────────────
  ids_rows, attn_rows, t_rows, y_rows = [], [], [], []
  n_valid_traj = 0
  n_total_traj = 0
  n_novel_traj = 0
  pbar = tqdm(total=num_traj, desc='[build-rollout] valid trajectories')
  b = 0
  while n_valid_traj < num_traj and b < max_batches:
    x0, snaps, t_vals = rollout_batch(model, gen_bs, length, steps, eps)
    decoded = tokenizer.batch_decode(x0)
    n_total_traj += len(decoded)
    # snaps: (steps, B, L). t_vals: list length steps.
    for j, smi in enumerate(decoded):
      is_valid, novel = _terminal_label(smi, qm9_set)
      if not is_valid:
        continue
      # Sample `snaps_per_traj` step indices for this trajectory (eq.(3)'s E_k;
      # uniform over steps is an unbiased estimate). Without replacement when
      # possible, to spread coverage across noise levels.
      k = min(snaps_per_traj, steps)
      step_idx = rng.choice(steps, size=k, replace=(steps < snaps_per_traj))
      for si in step_idx:
        xt_row = snaps[si, j].tolist()
        ids_rows.append(xt_row)
        attn_rows.append([1] * length)
        t_rows.append(float(t_vals[si]))
        y_rows.append(int(novel))
      n_valid_traj += 1
      n_novel_traj += int(novel)
      pbar.update(1)
      if n_valid_traj >= num_traj:
        break
    b += 1
  pbar.close()

  n_rows = len(ids_rows)
  validity = n_valid_traj / max(n_total_traj, 1)
  pos_frac = (sum(y_rows) / n_rows) if n_rows else 0.0
  print(f'[build-rollout] valid trajectories={n_valid_traj:,} '
        f'(from {n_total_traj:,} rolled out, validity≈{validity:.3f}); '
        f'terminal novel={n_novel_traj:,} '
        f'(traj novel_frac={n_novel_traj / max(n_valid_traj, 1):.3f})')
  print(f'[build-rollout] rows={n_rows:,} (snapshots) | '
        f'novel(1) frac={pos_frac:.3f}; steps/traj snapshots={snaps_per_traj}')

  # ── shuffle + train/val split + save ───────────────────────────────────────
  ds = datasets.Dataset.from_dict({
    'input_ids': ids_rows,
    'attention_mask': attn_rows,
    't': t_rows,
    'novel': y_rows,
  })
  ds = ds.shuffle(seed=seed)
  split = ds.train_test_split(test_size=val_frac, seed=seed)
  dd = datasets.DatasetDict(
    {'train': split['train'], 'validation': split['test']})
  os.makedirs(os.path.dirname(out_dir) or '.', exist_ok=True)
  dd.save_to_disk(out_dir)
  print(f'[build-rollout] saved {{train:{len(dd["train"]):,}, '
        f'validation:{len(dd["validation"]):,}}} -> {out_dir}')


if __name__ == '__main__':
  main()
