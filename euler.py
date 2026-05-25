"""Euler sampler for masked diffusion models (CTMC first-order discretization).

Implements the Euler method from Equation (7):

  x^i_{t_{k+1}} = a,       w.p. R̂^i_k(x^i_{t_k}, a)(t_{k+1} - t_k),  ∀a ≠ x^i_{t_k}
  x^i_{t_{k+1}} = x^i_{t_k}, w.p. 1 + R̂^i_k(x^i_{t_k}, x^i_{t_k})(t_{k+1} - t_k)

For SUBS / absorbing-state diffusion the token-wise reverse rate at a masked position is:
  R̂^i_t(mask → a) = p_θ(x^i=a | x_t) · σ'(t) / expm1(σ_t)

so the Euler step from t to s = t - Δt becomes:
  p(x^i_s = a    | x^i_t = mask) = p_θ(a|x_t) · dσ / expm1(σ_t)
  p(x^i_s = mask | x^i_t = mask) = 1 - dσ / expm1(σ_t)

where dσ = σ_t - σ_s > 0.  Unmasked positions are always copied over.
"""
import itertools
import json
import os
import typing

import datasets
import hydra
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

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
omegaconf.OmegaConf.register_new_resolver(
  'if_then_else', lambda condition, x, y: x if condition else y)


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
def euler_sample(
    model: diffusion.Diffusion,
    batch_size: typing.Optional[int] = None,
    length: typing.Optional[int] = None,
    num_steps: typing.Optional[int] = None,
    eps: float = 1e-5,
    disable_ema: typing.Optional[bool] = None,
    show_progress: bool = True) -> torch.Tensor:
  """Generate samples using the Euler (CTMC first-order) sampler.

  Args:
    model: Loaded ``Diffusion`` model (SUBS parameterization, loglinear noise).
    batch_size: Number of sequences per batch. Defaults to config value.
    length: Sequence length. Defaults to config value.
    num_steps: Number of discretisation steps N. Defaults to config value.
    eps: Small time offset to avoid σ → 0 at t = 0.
    disable_ema: Whether to skip EMA parameter swap.
    show_progress: Show tqdm progress bar.

  Returns:
    Integer tensor of shape ``(batch_size, length)``.
  """
  assert getattr(model.config.noise, 'type', None) == 'loglinear', (
    'Euler sampler requires a loglinear noise schedule.')

  batch_size = batch_size or model.config.loader.eval_batch_size
  length = length or model.config.model.length
  num_steps = num_steps or model.config.sampling.steps
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
    timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)

    iterator = range(num_steps)
    if show_progress:
      iterator = tqdm(iterator, total=num_steps, desc='Euler', leave=False)

    for i in iterator:
      t = timesteps[i] * torch.ones(batch_size, 1, device=model.device)
      s = timesteps[i + 1] * torch.ones(batch_size, 1, device=model.device)

      sigma_t, _ = model.noise(t)
      sigma_s, _ = model.noise(s)
      if sigma_t.ndim > 1:
        sigma_t = sigma_t.squeeze(-1)
      if sigma_s.ndim > 1:
        sigma_s = sigma_s.squeeze(-1)
      assert sigma_t.ndim == 1

      dsigma = sigma_t - sigma_s           # (B,)  positive
      scale = dsigma / torch.expm1(sigma_t)  # (B,)  transition mass per token

      # Model output: log p_θ(x_0 | x_t) for SUBS.
      # p_x0[..., mask_index] = 0 (set to -inf by _subs_parameterization).
      log_p_x0 = model.forward(x, sigma_t)   # (B, L, V)
      p_x0 = log_p_x0.exp()

      # Euler transition probabilities for each position:
      #   p(→ a)    = p_θ(a|x_t) * scale     for a ≠ mask
      #   p(→ mask) = 1 - scale
      trans_probs = p_x0 * scale[:, None, None]          # (B, L, V)
      trans_probs[..., model.mask_index] = (
        (1.0 - scale).clamp_min(0.0)[:, None]             # (B, L)
        .expand_as(trans_probs[..., model.mask_index]))

      _x = _sample_categorical(trans_probs)

      # Unmasked positions are never changed.
      copy_flag = (x != model.mask_index).to(x.dtype)
      x = copy_flag * x + (1 - copy_flag) * _x

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


@hydra.main(version_base=None, config_path='configs', config_name='config')
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
  for _ in tqdm(range(config.sampling.num_sample_batches),
                desc='Gen. Euler batches', leave=False):
    sample = euler_sample(model)
    samples.extend(tokenizer.batch_decode(sample))

  # --- qm9 evaluation ---
  label_col = omegaconf.OmegaConf.select(config, 'data.label_col', default=None)
  results_csv_path = omegaconf.OmegaConf.select(
    config, 'eval.results_csv_path', default=None)

  if label_col is not None:
    qm9_dataset = datasets.load_dataset(
      'yairschiff/qm9', trust_remote_code=True, split='train')
    mol_property_fn = _get_mol_property_fn(label_col)
    pctile = omegaconf.OmegaConf.select(config, 'data.label_col_pctile', default=90)
    pctile_val = np.percentile(qm9_dataset[label_col], q=pctile)

    print(f"QM9 reference  {label_col.upper()} Mean: "
          f"{np.mean(qm9_dataset[label_col]):.3f}, "
          f"Median: {np.median(qm9_dataset[label_col]):.3f}")
    print(f"Above {pctile}%ile threshold: {pctile_val:.3f}")

    valids, mol_property = [], []
    for t in samples:
      t = t.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '')
      try:
        mol = rdChem.MolFromSmiles(t)
        if mol is not None and len(t) > 0:
          valids.append(t)
          mol_property.append(mol_property_fn(mol))
      except rdkit.Chem.rdchem.KekulizeException:
        pass

    novel_set = set(valids) - set(qm9_dataset['canonical_smiles'])
    valid_pct = len(valids) / len(samples) if samples else 0.
    unique_pct = len(set(valids)) / len(valids) if valids else 0.
    novel_pct = len(novel_set) / len(valids) if valids else 0.
    mol_property_novel = [mol_property_fn(rdChem.MolFromSmiles(s)) for s in novel_set]

    print(f"\nEuler results (N={len(samples)}, steps={config.sampling.steps}):")
    print(f"  Valid:  {len(valids):,} / {len(samples):,} ({100*valid_pct:.2f}%)")
    print(f"  Unique (of valid): {len(set(valids)):,} ({100*unique_pct:.2f}%)")
    print(f"  Novel  (of valid): {len(novel_set):,} ({100*novel_pct:.2f}%)")
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
        'Seed': -1, 'Sampler': 'reference', 'Num Samples': len(qm9_dataset),
        'Valid': 1., 'Unique': 1., 'Novel': 1.,
        f'{label_col.upper()} Mean': np.mean(qm9_dataset[label_col]),
        f'{label_col.upper()} Median': np.median(qm9_dataset[label_col]),
        f'Novel {label_col.upper()} Mean': np.mean(qm9_dataset[label_col]),
        f'Novel {label_col.upper()} Median': np.median(qm9_dataset[label_col]),
      }
      euler_row = {
        'Seed': config.seed, 'Sampler': f'euler-{config.sampling.steps}',
        'Num Samples': len(samples),
        'Valid': valid_pct, 'Unique': unique_pct, 'Novel': novel_pct,
        f'{label_col.upper()} Mean': np.mean(mol_property) if mol_property else 0.,
        f'{label_col.upper()} Median': np.median(mol_property) if mol_property else 0.,
        f'Novel {label_col.upper()} Mean': np.mean(mol_property_novel) if mol_property_novel else 0.,
        f'Novel {label_col.upper()} Median': np.median(mol_property_novel) if mol_property_novel else 0.,
      }
      pd.DataFrame([ref_row, euler_row]).to_csv(results_csv_path, index=False)
      print(f"\nResults saved to {results_csv_path}")

  else:
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
