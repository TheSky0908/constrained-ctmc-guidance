"""Train discriminator h_φ(x_t, t) ≈ P(Y_T ∈ S | Y_t) for novelty-constrained FHS.

Implements the martingale MSE loss (Eq 3.15):
  min_φ E_{[0,T]} [∫_0^T (h_φ(t, Y_t) - 1(Y_T ∈ S))^2 dt]

Y_t are FHS trajectories from the frozen pretrained diffusion model.
S is the set of novel molecules (not in QM9 training set).
Training is purely off-policy.

Architecture of h_φ:
  - Frozen pretrained DIT backbone (reuses learned representations)
  - Trainable linear head: mean-pool over sequence → scalar → sigmoid

Usage:
  python -u discriminator_train.py \
    data=qm9 model=small backbone=dit model.length=32 \
    training.guidance=null parameterization=subs diffusion=absorbing_state \
    time_conditioning=False T=0 \
    eval.checkpoint_path=<ckpt> \
    discriminator.save_dir=<output_dir>
"""
import itertools
import os

import datasets
import hydra
import lightning as L
import omegaconf
import rdkit
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem as rdChem
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


@torch.no_grad()
def generate_trajectory(
    model: diffusion.Diffusion,
    batch_size: int,
    length: int,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
  """Run one FHS trajectory, recording (x_t, sigma_t) at every step.

  Returns:
    states: list of length L+1. states[0] = (all-mask, sigma(1));
            states[-1] = (clean, sigma~0).
    final_x: (B, L) fully denoised token tensor.
  """
  x = model._sample_prior(batch_size, length).to(model.device)
  t = torch.ones(batch_size, device=model.device)

  sigma_t, _ = model.noise(t)
  if sigma_t.ndim > 1:
    sigma_t = sigma_t.squeeze(-1)
  states = [(x.clone(), sigma_t.clone())]

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

    coordinate = torch.arange(length, device=x.device).unsqueeze(0).expand(batch_size, -1)
    masked_pos = torch.masked_select(coordinate, x == model.mask_index).reshape(batch_size, -1)
    rand_idx = torch.randint(high=num_masked, size=(batch_size,), device=x.device)
    selected = masked_pos[torch.arange(batch_size, device=x.device), rand_idx]

    token_probs = p_x0[torch.arange(batch_size, device=x.device), selected]
    new_tokens = _sample_categorical(token_probs)
    x[torch.arange(batch_size, device=x.device), selected] = new_tokens

    t = s
    states.append((x.clone(), sigma_s.clone()))

  return states, x


def get_novelty_labels(
    final_x: torch.Tensor,
    tokenizer,
    qm9_smiles_set: set,
) -> torch.Tensor:
  """Label = 1.0 if novel (not in QM9), else 0.0. Shape: (B,)."""
  decoded = tokenizer.batch_decode(final_x.cpu())
  labels = []
  for s in decoded:
    s = s.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '').strip()
    try:
      mol = rdChem.MolFromSmiles(s)
      if mol is not None and len(s) > 0:
        canonical = rdChem.MolToSmiles(mol)
        labels.append(0.0 if canonical in qm9_smiles_set else 1.0)
      else:
        labels.append(0.0)
    except Exception:
      labels.append(0.0)
  return torch.tensor(labels, dtype=torch.float32, device=final_x.device)


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config: omegaconf.DictConfig) -> None:
  L.seed_everything(config.seed)

  disc_cfg = config.discriminator
  save_dir = disc_cfg.save_dir
  num_steps = int(disc_cfg.num_steps)
  batch_size = int(disc_cfg.batch_size)
  lr = float(disc_cfg.lr)
  log_every = int(disc_cfg.log_every)
  save_every = int(disc_cfg.save_every)
  steps_per_traj = int(disc_cfg.steps_per_traj)

  os.makedirs(save_dir, exist_ok=True)

  # ── QM9 novelty reference set ──────────────────────────────────────────────
  print('Loading QM9 dataset...')
  qm9_dataset = datasets.load_dataset(
    'yairschiff/qm9', trust_remote_code=True, split='train')
  qm9_smiles_set = set(qm9_dataset['canonical_smiles'])
  print(f'  QM9 canonical SMILES: {len(qm9_smiles_set):,}')

  # ── Pretrained diffusion model (frozen) ────────────────────────────────────
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

  length = config.model.length
  hidden_size = config.model.hidden_size

  # ── h_φ: frozen backbone + trainable linear head ───────────────────────────
  # The backbone already encodes (x_t, sigma_t) into rich hidden representations.
  # We only train a single linear layer: mean-pool(hidden) → scalar → sigmoid.
  head = nn.Linear(hidden_size, 1).to(diff_model.device)
  optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

  print(f'Head parameters: {sum(p.numel() for p in head.parameters())}')
  print(f'Training for {num_steps} steps, batch_size={batch_size}')

  # ── Training loop ──────────────────────────────────────────────────────────
  loss_accum = 0.0
  novel_rate_accum = 0.0

  for step in tqdm(range(1, num_steps + 1), desc='Discriminator training'):
    # Off-policy trajectory from the frozen diffusion model.
    with torch.no_grad():
      states, final_x = generate_trajectory(diff_model, batch_size, length)
      labels = get_novelty_labels(final_x, tokenizer, qm9_smiles_set)  # (B,)
      novel_rate_accum += labels.mean().item()

    # Optionally subsample timesteps per trajectory to reduce compute.
    if steps_per_traj and steps_per_traj < len(states):
      indices = torch.randperm(len(states))[:steps_per_traj].tolist()
      states_to_use = [states[i] for i in sorted(indices)]
    else:
      states_to_use = states

    # Martingale MSE loss: mean over timesteps of (h_φ(x_t, t) - label)^2.
    head.train()
    optimizer.zero_grad()

    total_loss = torch.tensor(0.0, device=diff_model.device)
    for x_t, sigma_t in states_to_use:
      # Get hidden states from the frozen backbone (no gradient through backbone).
      with torch.no_grad():
        _, hidden_states = diff_model.backbone(x_t, sigma_t, return_hidden_states=True)
        last_hidden = hidden_states[-1].float()  # (B, L, hidden_size)

      # Mean-pool and apply trainable head.
      logit = head(last_hidden.mean(dim=1))     # (B, 1)
      h = torch.sigmoid(logit).squeeze(-1)      # (B,)
      total_loss = total_loss + F.mse_loss(h, labels)

    loss = total_loss / len(states_to_use)
    loss.backward()
    nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    optimizer.step()

    loss_accum += loss.item()

    if step % log_every == 0:
      avg_loss = loss_accum / log_every
      avg_novel = novel_rate_accum / log_every
      print(f'  step {step:>6d} | loss {avg_loss:.4f} | novel_rate {avg_novel:.3f}')
      loss_accum = 0.0
      novel_rate_accum = 0.0

    if step % save_every == 0:
      ckpt = {
        'step': step,
        'head_state_dict': head.state_dict(),
        'optimizer': optimizer.state_dict(),
        'hidden_size': hidden_size,
        'config': omegaconf.OmegaConf.to_container(config, resolve=True),
      }
      path = os.path.join(save_dir, f'h_phi_step{step:07d}.pt')
      torch.save(ckpt, path)
      print(f'  Saved: {path}')

  # Final checkpoint.
  ckpt = {
    'step': num_steps,
    'head_state_dict': head.state_dict(),
    'optimizer': optimizer.state_dict(),
    'hidden_size': hidden_size,
    'config': omegaconf.OmegaConf.to_container(config, resolve=True),
  }
  torch.save(ckpt, os.path.join(save_dir, 'h_phi_final.pt'))
  print(f'Training complete. Saved to {save_dir}')

  if not getattr(config.eval, 'disable_ema', True) and diff_model.ema is not None:
    diff_model.ema.restore(
      itertools.chain(diff_model.backbone.parameters(), diff_model.noise.parameters()))


if __name__ == '__main__':
  main()
