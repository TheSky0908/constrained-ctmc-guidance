"""Constrained DDPM sampler with h_φ-twisted rate matrix.

Same h/h twist as constrained_euler.py but applied to the DDPM posterior:
  q^λ(x^i_s = a | x^i_t = mask) ∝ q(x^i_s = a | x^i_t = mask) · h_s(x^{-i}⊕a) / h_t(x_t)

where the unconstrained DDPM posterior (absorbing state) is:
  q(a | mask) = p_θ(a | x_t) · (move_t - move_s) / move_t    for a ≠ mask
  q(mask | mask) = move_s / move_t

Hard-constraint enforcement: candidates with h_s(x^{-i}⊕a) < ``epsilon`` have
their twisted q_xs set to 0, so that mass naturally flows to the "stay at mask"
event for that position. No abort: positions that never see a candidate above
the threshold remain masked and will be decoded as invalid SMILES.

Usage:
  python -u constrained_ddpm.py \
    data=qm9 model=small backbone=dit model.length=32 \
    training.guidance=null parameterization=subs diffusion=absorbing_state \
    time_conditioning=False T=0 \
    eval.checkpoint_path=<diffusion_ckpt> \
    discriminator.checkpoint_path=<h_phi_final.pt> \
    sampling.steps=32 sampling.num_sample_batches=4 \
    loader.eval_global_batch_size=4
"""
import itertools
import json
import os

import datasets
import hydra
import lightning as L
import numpy as np
import omegaconf
import pandas as pd
import rdkit
import torch
import torch.nn as nn
import torch.nn.functional as F
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
  gumbel_norm = -(
    torch.rand(
      *categorical_probs.shape,
      device=categorical_probs.device,
      dtype=torch.float32,
    )
  ).log()
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _epsilon_suffix(path: str, epsilon: float) -> str:
  """Insert ``_eps{epsilon}`` before the file extension.

  e.g. 'foo/bar.json' with epsilon=0.05 → 'foo/bar_eps0.05.json'.
  Returns path unchanged if it is falsy.
  """
  if not path:
    return path
  base, ext = os.path.splitext(path)
  return f"{base}_eps{epsilon:g}{ext}"


def _load_head(checkpoint_path: str, device: torch.device) -> nn.Linear:
  ckpt = torch.load(checkpoint_path, map_location=device)
  head = nn.Linear(ckpt['hidden_size'], 1).to(device)
  head.load_state_dict(ckpt['head_state_dict'])
  head.eval()
  for p in head.parameters():
    p.requires_grad_(False)
  return head


def _h(backbone: nn.Module, head: nn.Linear,
       x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
  """h_φ(x, sigma) = sigmoid(head(mean_pool(backbone_hidden(x, sigma)))).

  Args:
    x:     (N, L) token indices
    sigma: (N,)   noise level (1-D)
  Returns:
    (N,) probabilities in [0, 1]
  """
  _, hidden_states = backbone(x, sigma, return_hidden_states=True)
  last_hidden = hidden_states[-1].float()       # (N, L, hidden_size)
  return torch.sigmoid(head(last_hidden.mean(dim=1))).squeeze(-1)  # (N,)


@torch.no_grad()
def constrained_ddpm_sample(
    model: diffusion.Diffusion,
    head: nn.Linear,
    batch_size: int,
    length: int,
    num_steps: int,
    epsilon: float = 0.05,
    eps: float = 1e-5,
    show_progress: bool = True,
) -> torch.Tensor:
  """DDPM sampler with h_φ-twisted posterior and hard h_φ threshold.

  For each masked position i, token a is sampled with probability:
    p^λ(a) ∝ p_θ(a | x_t) · (move_t - move_s) / move_t · h_s(x^{-i}⊕a) / h_t(x_t),
  with candidates whose h_s < ``epsilon`` zeroed out (mass flows to stay-at-mask).
  """
  assert getattr(model.config.noise, 'type', None) == 'loglinear'

  vocab_size = model.vocab_size
  device = model.device
  backbone = model.backbone

  xt = model._sample_prior(batch_size, length).to(device)
  timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
  dt = (1 - eps) / num_steps

  iterator = range(num_steps)
  if show_progress:
    iterator = tqdm(iterator, total=num_steps, desc='Constrained DDPM', leave=False)

  log_x_theta_cache = None

  for i in iterator:
    t = timesteps[i] * torch.ones(batch_size, 1, device=device)

    sigma_t, _ = model.noise(t)
    sigma_s, _ = model.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)   # (B,)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)   # (B,)

    move_t = (1 - torch.exp(-sigma_t))[:, None, None]   # (B, 1, 1)
    move_s = (1 - torch.exp(-sigma_s))[:, None, None]

    # ── Unconstrained DDPM posterior ─────────────────────────────────────────
    if log_x_theta_cache is not None:
      log_x_theta = log_x_theta_cache
    else:
      log_x_theta = model.forward(xt, sigma_t)           # (B, L, V)
    x_theta = log_x_theta.exp()

    q_xs = x_theta * (move_t - move_s) / move_t         # (B, L, V)  a ≠ mask
    q_xs[:, :, model.mask_index] = (move_s / move_t).squeeze(-1)

    # ── h_t(x_t): denominator ────────────────────────────────────────────────
    h_t = _h(backbone, head, xt, sigma_t)                # (B,)

    # ── Twist by h_s(x^{-i}⊕a) / h_t(x_t) for each masked position ─────────
    non_mask = torch.ones(vocab_size, device=device, dtype=torch.bool)
    non_mask[model.mask_index] = False

    for pos in range(length):
      masked_idx = (xt[:, pos] == model.mask_index).nonzero(as_tuple=True)[0]
      n = masked_idx.numel()
      if n == 0:
        continue

      # Candidates: (n, V, L) → (n*V, L)
      x_base = xt[masked_idx]
      x_cands = x_base.unsqueeze(1).expand(-1, vocab_size, -1).clone()
      x_cands[:, :, pos] = torch.arange(vocab_size, device=device)
      x_cands_flat = x_cands.reshape(n * vocab_size, length)

      sigma_s_rep = (
        sigma_s[masked_idx].unsqueeze(1)
        .expand(-1, vocab_size)
        .reshape(-1))                                     # (n*V,)

      h_s_flat = _h(backbone, head, x_cands_flat, sigma_s_rep)   # (n*V,)
      h_s = h_s_flat.reshape(n, vocab_size)              # (n, V)

      # Hard threshold: drop candidates whose h_s < epsilon. Their q_xs
      # becomes 0, so the mass redistributes to "stay at mask".
      h_s = torch.where(h_s < epsilon, torch.zeros_like(h_s), h_s)

      ratio = h_s / h_t[masked_idx].unsqueeze(-1).clamp_min(1e-8)  # (n, V)

      q_xs[masked_idx[:, None], pos, non_mask] = (
        q_xs[masked_idx[:, None], pos, non_mask] * ratio[:, non_mask])

      non_mask_sum = q_xs[masked_idx, pos, :][:, non_mask].sum(dim=-1)
      q_xs[masked_idx, pos, model.mask_index] = (1.0 - non_mask_sum).clamp_min(0.0)

    # ── Sample ───────────────────────────────────────────────────────────────
    xs = _sample_categorical(q_xs)
    copy_flag = (xt != model.mask_index)
    xs = torch.where(copy_flag, xt, xs)

    # Cache: reuse log_x_theta next step only if state did not change.
    if model.config.sampling.use_cache and torch.allclose(xs, xt):
      log_x_theta_cache = log_x_theta
    else:
      log_x_theta_cache = None

    xt = xs

  return xt


def _get_mol_property_fn(prop: str):
  if prop == 'qed':
    return QED.qed
  if prop == 'ring_count':
    return lambda mol: len(rdChem.GetSymmSSSR(mol))
  raise NotImplementedError(f'Property function for {prop} not implemented')


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config: omegaconf.DictConfig) -> None:
  L.seed_everything(config.seed)

  checkpoint_path = config.discriminator.checkpoint_path
  if not checkpoint_path:
    raise ValueError('Set discriminator.checkpoint_path to the trained h_phi checkpoint')

  tokenizer = dataloader.get_tokenizer(config)
  diff_model = diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config,
    logger=False)
  diff_model.eval()
  if torch.cuda.is_available():
    diff_model = diff_model.to('cuda')
  for p in diff_model.parameters():
    p.requires_grad_(False)

  if not getattr(config.eval, 'disable_ema', True) and diff_model.ema is not None:
    diff_model.ema.store(
      itertools.chain(diff_model.backbone.parameters(), diff_model.noise.parameters()))
    diff_model.ema.copy_to(
      itertools.chain(diff_model.backbone.parameters(), diff_model.noise.parameters()))

  head = _load_head(checkpoint_path, diff_model.device)

  batch_size = config.sampling.batch_size
  length = config.model.length
  num_steps = config.sampling.steps
  epsilon = float(omegaconf.OmegaConf.select(
    config, 'discriminator.epsilon', default=0.05))
  samples_path = _epsilon_suffix(
    config.eval.generated_samples_path, epsilon)
  results_csv_path_tagged = _epsilon_suffix(
    omegaconf.OmegaConf.select(config, 'eval.results_csv_path', default=None),
    epsilon)

  # ── Generation ───────────────────────────────────────────────────────────────
  samples = []
  for _ in tqdm(range(config.sampling.num_sample_batches),
                desc='Constrained DDPM batches', leave=False):
    sample = constrained_ddpm_sample(
      diff_model, head, batch_size, length, num_steps, epsilon=epsilon)
    samples.extend(tokenizer.batch_decode(sample))

  # ── Evaluation ───────────────────────────────────────────────────────────────
  label_col = omegaconf.OmegaConf.select(config, 'data.label_col', default=None)
  results_csv_path = results_csv_path_tagged

  print('Loading QM9 for evaluation...')
  qm9_dataset = datasets.load_dataset(
    'yairschiff/qm9', trust_remote_code=True, split='train')
  qm9_smiles_set = set(qm9_dataset['canonical_smiles'])

  mol_property_fn = _get_mol_property_fn(label_col) if label_col else None

  valids, mol_property = [], []
  for t in samples:
    t = t.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '').strip()
    try:
      mol = rdChem.MolFromSmiles(t)
      if mol is not None and len(t) > 0:
        valids.append(t)
        if mol_property_fn is not None:
          mol_property.append(mol_property_fn(mol))
    except rdkit.Chem.rdchem.KekulizeException:
      pass

  novel_set = set(valids) - qm9_smiles_set
  n_total  = len(samples)
  n_valid  = len(valids)
  n_unique = len(set(valids))
  n_novel  = len(novel_set)

  valid_pct       = n_valid  / n_total  if n_total else 0.
  unique_pct      = n_unique / n_valid  if n_valid else 0.
  novel_pct       = n_novel  / n_valid  if n_valid else 0.
  constraint_rate = n_novel  / n_total  if n_total else 0.

  mol_property_novel = (
    [mol_property_fn(rdChem.MolFromSmiles(s)) for s in novel_set]
    if mol_property_fn else [])

  if label_col:
    pctile = omegaconf.OmegaConf.select(config, 'data.label_col_pctile', default=90)
    pctile_val = np.percentile(qm9_dataset[label_col], q=pctile)
    print(f"QM9 reference  {label_col.upper()} Mean: "
          f"{np.mean(qm9_dataset[label_col]):.3f}, "
          f"Median: {np.median(qm9_dataset[label_col]):.3f}")
    print(f"Above {pctile}%ile threshold: {pctile_val:.3f}")

  print(f"\nConstrained DDPM results (N={n_total}, steps={num_steps}, epsilon={epsilon:g}):")
  print(f"  Valid:                     {n_valid:,} / {n_total:,} ({100*valid_pct:.2f}%)")
  print(f"  Unique       (of valid):   {n_unique:,} / {n_valid:,} ({100*unique_pct:.2f}%)")
  print(f"  Novel        (of valid):   {n_novel:,} / {n_valid:,} ({100*novel_pct:.2f}%)")
  print(f"  Constraint satisfaction:   {n_novel:,} / {n_total:,} ({100*constraint_rate:.2f}%)")
  if mol_property:
    print(f"  {label_col.upper()} Mean: {np.mean(mol_property):.3f}, "
          f"Median: {np.median(mol_property):.3f}")
  if mol_property_novel:
    print(f"  Novel {label_col.upper()} Mean: {np.mean(mol_property_novel):.3f}, "
          f"Median: {np.median(mol_property_novel):.3f}")

  if samples_path:
    out_dir = os.path.dirname(samples_path)
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)
    payload = {
      'epsilon': epsilon,
      'valid': valids,
      'novel': list(novel_set),
      'constraint_satisfaction_rate': constraint_rate,
      'valid_pct': valid_pct,
      'unique_pct': unique_pct,
      'novel_pct': novel_pct,
    }
    if label_col:
      payload[f'{label_col}_valid'] = mol_property
      payload[f'{label_col}_novel'] = mol_property_novel
    with open(samples_path, 'w') as f:
      json.dump(payload, f, indent=4)

  if results_csv_path:
    row = {
      'Seed': config.seed,
      'Sampler': f'constrained-ddpm-{num_steps}-eps{epsilon:g}',
      'Epsilon': epsilon,
      'Num Samples': n_total,
      'Valid': valid_pct,
      'Unique': unique_pct,
      'Novel (of valid)': novel_pct,
      'Constraint Satisfaction Rate': constraint_rate,
    }
    if label_col:
      row.update({
        f'{label_col.upper()} Mean': np.mean(mol_property) if mol_property else 0.,
        f'{label_col.upper()} Median': np.median(mol_property) if mol_property else 0.,
        f'Novel {label_col.upper()} Mean': np.mean(mol_property_novel) if mol_property_novel else 0.,
        f'Novel {label_col.upper()} Median': np.median(mol_property_novel) if mol_property_novel else 0.,
      })
    pd.DataFrame([row]).to_csv(results_csv_path, index=False)
    print(f"\nResults saved to {results_csv_path}")

  if not getattr(config.eval, 'disable_ema', True) and diff_model.ema is not None:
    diff_model.ema.restore(
      itertools.chain(diff_model.backbone.parameters(), diff_model.noise.parameters()))


if __name__ == '__main__':
  main()
