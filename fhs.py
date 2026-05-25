"""First-hitting sampler for masked diffusion models.

This module implements the basic first-order first-hitting sampler (FHS)
for absorbing-state / masked diffusion models. One masked position is
filled per step: the next denoising time is drawn as s = u^(1/n) * t
where u ~ Uniform(0,1) and n is the current number of masked tokens.
"""
import json
import os
import typing

import datasets
import hydra
import itertools
import lightning as L
import numpy as np
import omegaconf
import pandas as pd
import rdkit
import torch
from rdkit import Chem as rdChem
from rdkit.Chem import QED
from tqdm.auto import tqdm

import dataloader
import diffusion

rdkit.rdBase.DisableLog('rdApp.error')


omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)
omegaconf.OmegaConf.register_new_resolver(
  'if_then_else',
  lambda condition, x, y: x if condition else y
)


def _sample_categorical(categorical_probs: torch.Tensor) -> torch.Tensor:
  """Gumbel-max trick on a probability tensor (not log-probs)."""
  gumbel_norm = -(
    torch.rand(
      *categorical_probs.shape,
      device=categorical_probs.device,
      dtype=torch.float32,
    )
  ).log()
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


@torch.no_grad()
def first_hitting_sample(
    model: diffusion.Diffusion,
    batch_size: typing.Optional[int] = None,
    length: typing.Optional[int] = None,
    disable_ema: typing.Optional[bool] = None,
    show_progress: bool = True) -> torch.Tensor:
  """Generate samples using first-hitting sampling (basic first order).

  At each of the L steps exactly one masked position is unmasked. The
  next conditioning time is s = u^(1/n) * t where n is the number of
  still-masked tokens and u ~ Uniform(0,1). The model is queried at s,
  not at t, mirroring ``restore_model_and_sample_first_hitting`` in
  diffusion.py.

  Args:
    model: Loaded ``Diffusion`` model.
    batch_size: Number of sequences to sample. Defaults to config value.
    length: Sequence length. Defaults to config value.
    disable_ema: If False, temporarily swap in EMA params when available.
    show_progress: Whether to show a tqdm progress bar.

  Returns:
    Integer tensor of shape ``(batch_size, length)``.
  """
  assert getattr(model.config.noise, 'type', None) == 'loglinear', (
    'First-hitting sampling requires a loglinear noise schedule.')

  batch_size = batch_size or model.config.loader.eval_batch_size
  length = length or model.config.model.length
  if disable_ema is None:
    disable_ema = getattr(model.config.eval, 'disable_ema', True)

  if not disable_ema and model.ema is not None:
    model.ema.store(
      itertools.chain(model.backbone.parameters(), model.noise.parameters()))
    model.ema.copy_to(
      itertools.chain(model.backbone.parameters(), model.noise.parameters()))
  model.backbone.eval()
  model.noise.eval()

  try:
    x = model._sample_prior(batch_size, length).to(model.device)
    t = torch.ones(batch_size, device=model.device)

    iterator = range(length)
    if show_progress:
      iterator = tqdm(iterator, total=length, desc='FHS', leave=False)

    for i in iterator:
      num_masked = length - i

      u = torch.float_power(torch.rand_like(t), 1.0 / num_masked)
      s = (u * t).to(t.dtype)

      sigma_s, _ = model.noise(s)
      p_x0 = model.forward(x, sigma_s).exp()
      p_x0[..., model.mask_index] = 0.0
      p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True).clamp_min(1e-12)

      # Pick one masked position uniformly at random per sequence.
      # At step i all sequences have exactly num_masked masked tokens.
      coordinate = torch.arange(length, device=x.device).unsqueeze(0).expand(batch_size, -1)
      masked_pos = torch.masked_select(coordinate, x == model.mask_index).reshape(batch_size, -1)
      rand_idx = torch.randint(high=num_masked, size=(batch_size,), device=x.device)
      selected = masked_pos[torch.arange(batch_size, device=x.device), rand_idx]

      token_probs = p_x0[torch.arange(batch_size, device=x.device), selected]
      new_tokens = _sample_categorical(token_probs)
      x[torch.arange(batch_size, device=x.device), selected] = new_tokens

      t = s

    return x
  finally:
    if not disable_ema and model.ema is not None:
      model.ema.restore(
        itertools.chain(model.backbone.parameters(), model.noise.parameters()))
    model.backbone.train()
    model.noise.train()


def _get_mol_property_fn(prop: str):
  if prop == 'qed':
    return QED.qed
  if prop == 'ring_count':
    return lambda mol: len(rdChem.GetSymmSSSR(mol))
  raise NotImplementedError(f'Property function for {prop} not implemented')


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config: omegaconf.DictConfig) -> None:
  L.seed_everything(config.seed)

  tokenizer = dataloader.get_tokenizer(config)
  model = diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config,
    logger=False)
  model.eval()
  if torch.cuda.is_available():
    model = model.to('cuda')

  # --- generation ---
  samples = []
  for _ in tqdm(
      range(config.sampling.num_sample_batches),
      desc='Gen. FHS batches',
      leave=False):
    sample = first_hitting_sample(model)
    samples.extend(tokenizer.batch_decode(sample))

  # --- qm9 evaluation (mirrors qm9_eval.py) ---
  label_col = omegaconf.OmegaConf.select(config, 'data.label_col', default=None)
  results_csv_path = omegaconf.OmegaConf.select(
    config, 'eval.results_csv_path', default=None)

  if label_col is not None:
    qm9_dataset = datasets.load_dataset(
      'yairschiff/qm9', trust_remote_code=True, split='train')
    mol_property_fn = _get_mol_property_fn(label_col)
    pctile = omegaconf.OmegaConf.select(
      config, 'data.label_col_pctile', default=90)
    pctile_val = np.percentile(qm9_dataset[label_col], q=pctile)

    print(f"QM9 reference  - {label_col.upper()} Mean: {np.mean(qm9_dataset[label_col]):.3f}, "
          f"Median: {np.median(qm9_dataset[label_col]):.3f}")
    print(f"Above {pctile}%ile threshold: {pctile_val:.3f}")

    valids, invalids, mol_property = [], [], []
    for t in samples:
      t = t.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '')
      try:
        mol = rdChem.MolFromSmiles(t)
        if mol is None or len(t) == 0:
          invalids.append(t)
        else:
          valids.append(t)
          mol_property.append(mol_property_fn(mol))
      except rdkit.Chem.rdchem.KekulizeException:
        invalids.append(t)

    valid_pct = len(valids) / len(samples) if samples else 0.
    unique = len(set(valids))
    novel_set = set(valids) - set(qm9_dataset['canonical_smiles'])
    novel = len(novel_set)
    unique_pct = unique / len(valids) if valids else 0.
    novel_pct = novel / len(valids) if valids else 0.
    mol_property_novel = [
      mol_property_fn(rdChem.MolFromSmiles(s)) for s in novel_set]

    print(f"\nFHS results (N={len(samples)}):")
    print(f"  Valid:  {len(valids):,} / {len(samples):,} ({100*valid_pct:.2f}%)")
    print(f"  Unique (of valid): {unique:,} / {len(valids):,} ({100*unique_pct:.2f}%)")
    print(f"  Novel  (of valid): {novel:,} / {len(valids):,} ({100*novel_pct:.2f}%)")
    if mol_property:
      print(f"  {label_col.upper()} Mean: {np.mean(mol_property):.3f}, "
            f"Median: {np.median(mol_property):.3f}")
    if mol_property_novel:
      print(f"  Novel {label_col.upper()} Mean: {np.mean(mol_property_novel):.3f}, "
            f"Median: {np.median(mol_property_novel):.3f}")

    if config.eval.generated_samples_path:
      out_dir = os.path.dirname(config.eval.generated_samples_path)
      if out_dir:
        os.makedirs(out_dir, exist_ok=True)
      with open(config.eval.generated_samples_path, 'w') as f:
        json.dump({
          'valid': valids,
          'novel': list(novel_set),
          f'{label_col}_valid': mol_property,
          f'{label_col}_novel': mol_property_novel,
        }, f, indent=4)

    if results_csv_path:
      ref_row = {
        'Seed': -1, 'T': -1, 'Sampler': 'reference',
        'Num Samples': len(qm9_dataset),
        'Valid': 1.0, 'Unique': 1.0, 'Novel': 1.0,
        f'{label_col.upper()} Mean': np.mean(qm9_dataset[label_col]),
        f'{label_col.upper()} 25%ile': np.percentile(qm9_dataset[label_col], q=25),
        f'{label_col.upper()} Median': np.median(qm9_dataset[label_col]),
        f'{label_col.upper()} 75%ile': np.percentile(qm9_dataset[label_col], q=75),
        f'Novel {label_col.upper()} Mean': np.mean(qm9_dataset[label_col]),
        f'Novel {label_col.upper()} 25%ile': np.percentile(qm9_dataset[label_col], q=25),
        f'Novel {label_col.upper()} Median': np.median(qm9_dataset[label_col]),
        f'Novel {label_col.upper()} 75%ile': np.percentile(qm9_dataset[label_col], q=75),
      }
      fhs_row = {
        'Seed': config.seed, 'T': config.model.length, 'Sampler': 'fhs',
        'Num Samples': len(samples),
        'Valid': valid_pct, 'Unique': unique_pct, 'Novel': novel_pct,
        f'{label_col.upper()} Mean': np.mean(mol_property) if mol_property else 0.,
        f'{label_col.upper()} 25%ile': np.percentile(mol_property, q=25) if mol_property else 0.,
        f'{label_col.upper()} Median': np.median(mol_property) if mol_property else 0.,
        f'{label_col.upper()} 75%ile': np.percentile(mol_property, q=75) if mol_property else 0.,
        f'Novel {label_col.upper()} Mean': np.mean(mol_property_novel) if mol_property_novel else 0.,
        f'Novel {label_col.upper()} 25%ile': np.percentile(mol_property_novel, q=25) if mol_property_novel else 0.,
        f'Novel {label_col.upper()} Median': np.median(mol_property_novel) if mol_property_novel else 0.,
        f'Novel {label_col.upper()} 75%ile': np.percentile(mol_property_novel, q=75) if mol_property_novel else 0.,
      }
      pd.DataFrame([ref_row, fhs_row]).to_csv(results_csv_path, index=False)
      print(f"\nResults saved to {results_csv_path}")

  else:
    # no evaluation, just dump raw text
    if config.eval.generated_samples_path:
      out_dir = os.path.dirname(config.eval.generated_samples_path)
      if out_dir:
        os.makedirs(out_dir, exist_ok=True)
      with open(config.eval.generated_samples_path, 'w') as f:
        json.dump({'samples': samples}, f, indent=2)
    else:
      for s in samples:
        print(s)


if __name__ == '__main__':
  main()
