"""Build the on-disk `qm9_novel` dataset for the novelty D-CBG classifier.

The novelty guidance classifier learns p(novel | x_t) — given a (noised) token
state, the probability the final decoded SMILES is *absent* from the QM9 train
set. QM9-train alone cannot train this (every molecule is label-0 "not novel"),
so we construct labels offline, mirroring how text_toxicity built its
`jigsaw_scored` set (label_jigsaw_with_surrogate.py):

  1. Generate valid SMILES from the frozen base MDLM (one-token-per-step FHS,
     same sampler as discriminator_train.py).
  2. Canonicalize; label novel=1 if the canonical SMILES is NOT in QM9-train,
     else 0 (rediscovered).
  3. Mix in QM9-train molecules as guaranteed label-0 ("not novel") to balance
     the classes and feed the classifier the real training-set distribution.

Saved as a HF DatasetDict {train, validation} with columns
`canonical_smiles` + `novel`, loaded by the `qm9_novel` dataloader branch.

Run:
  python -u experiments/molecule_novelty/build_novelty_dataset.py \
    data=qm9 model=small backbone=dit model.length=32 \
    training.guidance=null parameterization=subs diffusion=absorbing_state \
    time_conditioning=False T=0 \
    eval.checkpoint_path=outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt \
    +build.num_valid=50000 +build.gen_batch_size=512 \
    +build.out_dir=.data_cache/qm9_novelty_scored
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


def _sample_categorical(categorical_probs: torch.Tensor) -> torch.Tensor:
  gumbel_norm = -(
    torch.rand(
      *categorical_probs.shape,
      device=categorical_probs.device,
      dtype=torch.float32,
    )
  ).log()
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


@torch.no_grad()
def generate_batch(model, batch_size: int, length: int) -> torch.Tensor:
  """One-token-per-step FHS generation (same kernel as discriminator_train).

  Returns final denoised token tensor (B, L)."""
  x = model._sample_prior(batch_size, length).to(model.device)
  t = torch.ones(batch_size, device=model.device)
  for i in range(length):
    num_masked = length - i
    u = torch.float_power(torch.rand_like(t), 1.0 / num_masked)
    s = (u * t).to(t.dtype)
    sigma_s, _ = model.noise(s)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    p_x0 = model.forward(x, sigma_s).exp()
    p_x0[..., model.mask_index] = 0.0
    p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    coordinate = torch.arange(length, device=x.device).unsqueeze(0).expand(
      batch_size, -1)
    masked_pos = torch.masked_select(
      coordinate, x == model.mask_index).reshape(batch_size, -1)
    rand_idx = torch.randint(
      high=num_masked, size=(batch_size,), device=x.device)
    selected = masked_pos[torch.arange(batch_size, device=x.device), rand_idx]
    token_probs = p_x0[torch.arange(batch_size, device=x.device), selected]
    new_tokens = _sample_categorical(token_probs)
    x[torch.arange(batch_size, device=x.device), selected] = new_tokens
    t = s
  return x


def _canon_valid(smiles_list, qm9_set):
  """Decode → validate → canonicalize. Returns list of (canonical, is_novel)."""
  out = []
  for s in smiles_list:
    s = s.replace('<bos>', '').replace('<eos>', '').replace(
      '<pad>', '').strip()
    if not s:
      continue
    try:
      mol = rdChem.MolFromSmiles(s)
    except Exception:
      mol = None
    if mol is None:
      continue
    canon = rdChem.MolToSmiles(mol)
    if not canon:
      continue
    out.append((canon, 0 if canon in qm9_set else 1))
  return out


@hydra.main(version_base=None, config_path='../../configs',
            config_name='config')
def main(config: omegaconf.DictConfig) -> None:
  L.seed_everything(config.seed)

  sel = lambda k, d: omegaconf.OmegaConf.select(config, k, default=d)
  num_valid = int(sel('build.num_valid', 50000))
  gen_bs = int(sel('build.gen_batch_size', 512))
  max_batches = int(sel('build.max_gen_batches', 100000))
  val_frac = float(sel('build.val_frac', 0.05))
  balance = bool(sel('build.balance', True))
  dedup = bool(sel('build.dedup', True))
  out_dir = str(sel('build.out_dir', '.data_cache/qm9_novelty_scored'))
  seed = int(config.seed)

  # ── QM9 novelty reference set ──────────────────────────────────────────────
  print('[build] loading QM9 train...')
  qm9 = datasets.load_dataset(
    'yairschiff/qm9', trust_remote_code=True, split='train')
  qm9_canon = qm9['canonical_smiles']
  qm9_set = set(qm9_canon)
  print(f'[build] QM9 canonical SMILES: {len(qm9_set):,}')

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

  # ── generate + label ───────────────────────────────────────────────────────
  gen = {}  # canonical -> novel label (dedup) ; or list if not dedup
  gen_list = []
  n_valid = 0
  n_total = 0
  pbar = tqdm(total=num_valid, desc='[build] gen valid')
  b = 0
  while n_valid < num_valid and b < max_batches:
    final_x = generate_batch(model, gen_bs, length)
    decoded = tokenizer.batch_decode(final_x.cpu())
    n_total += len(decoded)
    for canon, is_novel in _canon_valid(decoded, qm9_set):
      if dedup:
        if canon not in gen:
          gen[canon] = is_novel
          n_valid += 1
          pbar.update(1)
      else:
        gen_list.append((canon, is_novel))
        n_valid += 1
        pbar.update(1)
    b += 1
  pbar.close()

  if dedup:
    pairs = list(gen.items())
  else:
    pairs = gen_list
  gen_smiles = [c for c, _ in pairs]
  gen_labels = [l for _, l in pairs]
  n_gen = len(gen_smiles)
  n_gen_novel = int(sum(gen_labels))
  n_gen_redisc = n_gen - n_gen_novel
  validity = n_valid / max(n_total, 1)
  print(f'[build] generated valid={n_gen:,} (from {n_total:,} decoded, '
        f'validity≈{validity:.3f}); novel={n_gen_novel:,} '
        f'rediscovered={n_gen_redisc:,}')

  # ── mix QM9-train as guaranteed label-0 to balance classes ─────────────────
  smiles = list(gen_smiles)
  labels = list(gen_labels)
  if balance:
    n_mix = max(0, n_gen_novel - n_gen_redisc)
    n_mix = min(n_mix, len(qm9_canon))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(qm9_canon), size=n_mix, replace=False)
    for j in idx:
      smiles.append(qm9_canon[int(j)])
      labels.append(0)
    print(f'[build] mixed in {n_mix:,} QM9-train molecules as label-0')

  n0 = labels.count(0)
  n1 = labels.count(1)
  print(f'[build] final dataset: {len(smiles):,} rows | '
        f'not-novel(0)={n0:,} novel(1)={n1:,} | pos_frac={n1/len(labels):.3f}')

  # ── shuffle + train/val split + save ───────────────────────────────────────
  ds = datasets.Dataset.from_dict(
    {'canonical_smiles': smiles, 'novel': labels})
  ds = ds.shuffle(seed=seed)
  split = ds.train_test_split(test_size=val_frac, seed=seed)
  dd = datasets.DatasetDict(
    {'train': split['train'], 'validation': split['test']})
  os.makedirs(os.path.dirname(out_dir) or '.', exist_ok=True)
  dd.save_to_disk(out_dir)
  print(f'[build] saved {{train:{len(dd["train"]):,}, '
        f'validation:{len(dd["validation"]):,}}} -> {out_dir}')


if __name__ == '__main__':
  main()
