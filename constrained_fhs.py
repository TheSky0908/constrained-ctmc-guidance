"""Constrained FHS sampler with h_φ-twisted first-hitting distribution.

Same h/h twist as constrained_euler.py / constrained_ddpm.py but applied to
the First-Hitting Sampler (FHS).

Theory: for the h-twisted reverse process, the per-position unmasking rate
is r_j ∝ Σ_a μ^j(a|x,t)·h_s(x^{-j}⊕a) / h_t(x), and the inner sum equals
h_t(x) exactly, so r_j is the same across positions. Hence the next index
is **uniform over masked positions**, just like vanilla FHS; only the token
distribution at that position is twisted:

  p^λ(token=a | pos=j, x_t) ∝ p_θ(a | x_t, σ_s) · h_s(x^{-j}⊕a)

Hard-constraint enforcement: candidates with h_s(x^{-j}⊕a) < ``epsilon`` are
zeroed out before renormalisation. If every candidate at the selected
position is below the threshold for any sequence in the batch, a warning
is printed and the sampler raises to terminate the run.

Usage:
  python -u constrained_fhs.py \
    data=qm9 model=small backbone=dit model.length=32 \
    training.guidance=null parameterization=subs diffusion=absorbing_state \
    time_conditioning=False T=0 \
    eval.checkpoint_path=<diffusion_ckpt> \
    discriminator.checkpoint_path=<h_phi_final.pt> \
    sampling.num_sample_batches=4 \
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
    sigma: (N,)   noise level
  Returns:
    (N,) probabilities in [0, 1]
  """
  _, hidden_states = backbone(x, sigma, return_hidden_states=True)
  last_hidden = hidden_states[-1].float()       # (N, L, hidden_size)
  return torch.sigmoid(head(last_hidden.mean(dim=1))).squeeze(-1)  # (N,)


@torch.no_grad()
def constrained_fhs_sample(
    model: diffusion.Diffusion,
    head: nn.Linear,
    batch_size: int,
    length: int,
    epsilon: float = 0.05,
    show_progress: bool = True,
) -> torch.Tensor:
  """FHS sampler with h_φ-twisted token distribution and hard h_φ threshold.

  Per-step procedure:
    1. Sample one masked position uniformly at random (theory: per-position
       rate Σ_a μ·h equals h_t(x), so positions are equiprobable).
    2. At that position compute h_φ(x^{-j}⊕a, σ_s) for every non-mask token a.
    3. Zero out candidates with h_φ < ``epsilon``.
    4. Sample the next token from p_θ(a|x,σ_s)·h_φ(x^{-j}⊕a, σ_s) renormalised
       over the surviving candidates.

  Raises RuntimeError if every candidate at the selected position has
  h_φ < epsilon for any sequence in the batch.
  """
  assert getattr(model.config.noise, 'type', None) == 'loglinear'

  vocab_size = model.vocab_size
  device = model.device
  backbone = model.backbone
  batch_idx = torch.arange(batch_size, device=device)
  vocab_idx = torch.arange(vocab_size, device=device)

  x = model._sample_prior(batch_size, length).to(device)
  t = torch.ones(batch_size, device=device)

  iterator = range(length)
  if show_progress:
    iterator = tqdm(iterator, total=length, desc='Constrained FHS', leave=False)

  for step in iterator:
    num_masked = length - step

    # ── FHS time step ─────────────────────────────────────────────────────────
    u = torch.float_power(torch.rand_like(t), 1.0 / num_masked)
    s = (u * t).to(t.dtype)

    sigma_s, _ = model.noise(s)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)

    # ── Unconstrained p_x0 ────────────────────────────────────────────────────
    p_x0 = model.forward(x, sigma_s).exp()          # (B, L, V)
    p_x0[..., model.mask_index] = 0.0
    p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # ── Sample one masked position uniformly per sequence ────────────────────
    coordinate = torch.arange(length, device=device).unsqueeze(0).expand(batch_size, -1)
    masked_pos = torch.masked_select(
      coordinate, x == model.mask_index).reshape(batch_size, -1)  # (B, num_masked)
    rand_idx = torch.randint(high=num_masked, size=(batch_size,), device=device)
    selected_pos = masked_pos[batch_idx, rand_idx]               # (B,)

    # ── h_φ at every candidate (B·V forwards) ────────────────────────────────
    x_cands = x.unsqueeze(1).expand(-1, vocab_size, -1).clone()  # (B, V, L)
    x_cands[batch_idx[:, None], vocab_idx[None, :], selected_pos[:, None]] = (
      vocab_idx[None, :])
    x_cands_flat = x_cands.reshape(batch_size * vocab_size, length)
    sigma_s_rep = sigma_s.unsqueeze(1).expand(-1, vocab_size).reshape(-1)
    h_s = _h(backbone, head, x_cands_flat, sigma_s_rep).reshape(
      batch_size, vocab_size)                                    # (B, V)

    # ── Hard threshold: drop candidates whose h_φ is below epsilon ────────────
    h_s = torch.where(h_s < epsilon, torch.zeros_like(h_s), h_s)

    # ── Token weights: p_θ · h_φ ; mask token already zeroed in p_x0 ─────────
    tok_w = p_x0[batch_idx, selected_pos] * h_s                  # (B, V)
    tok_w[:, model.mask_index] = 0.0

    # Abort if any sequence has no surviving candidate at this position.
    no_survivor = (tok_w.sum(dim=-1) == 0)                       # (B,)
    if no_survivor.any():
      n_dead = int(no_survivor.sum().item())
      msg = (f'[Constrained FHS] step {step}: {n_dead}/{batch_size} sequence(s) '
             f'have no candidate token with h_phi >= epsilon={epsilon} at the '
             f'selected position. Terminating sampling.')
      print(msg)
      raise RuntimeError(msg)

    tok_w = tok_w / tok_w.sum(dim=-1, keepdim=True)
    new_tokens = _sample_categorical(tok_w)                      # (B,)

    x[batch_idx, selected_pos] = new_tokens
    t = s

  return x


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

  batch_size = config.loader.eval_batch_size
  length = config.model.length
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
                desc='Constrained FHS batches', leave=False):
    sample = constrained_fhs_sample(
      diff_model, head, batch_size, length, epsilon=epsilon)
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

  print(f"\nConstrained FHS results (N={n_total}, steps={length}, epsilon={epsilon:g}):")
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
      'Sampler': f'constrained-fhs-eps{epsilon:g}',
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
