"""SA-constrained QM9 evaluation for MDLM + D-CBG (gamma).

Reproduces CDD paper (NeurIPS 2025, arXiv:2503.09790) Fig 4 LEFT table row
"MDLM_D-CBG gamma=3" for synthetic accessibility constraint.

CDD paper reports for that row:
  Valid=436, Novel=14, QED=0.37, Viol@tau=3.0: 50.5%, 3.5: 48.6%,
  4.0: 46.1%, 4.5: 44.7% (paper's "Novel" = valid AND not-in-train
  AND SA <= 3.0; QED reported for those novel ones only).

Run:
  CUDA_VISIBLE_DEVICES=0 python sa_eval.py \
    data=qm9 model=small backbone=dit model.length=32 \
    parameterization=subs diffusion=absorbing_state \
    time_conditioning=False T=0 \
    eval.checkpoint_path=outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt \
    guidance=cbg \
    guidance.method=cbg guidance.gamma=3 guidance.condition=1 \
    guidance.classifier_checkpoint_path=<sa_classifier.ckpt> \
    sampling.steps=128 sampling.num_sample_batches=4 \
    loader.eval_global_batch_size=256
"""
import json
import os
import sys

import datasets
import hydra
import lightning as L
import numpy as np
import omegaconf
import pandas as pd
import rdkit
import torch
from rdkit import Chem as rdChem
from rdkit.Chem import QED, RDConfig
from tqdm.auto import tqdm

import dataloader
import diffusion

# RDKit Contrib sascorer (synthetic accessibility) — same scorer CDD uses for
# the external violation evaluator.
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer  # noqa: E402

rdkit.rdBase.DisableLog('rdApp.error')

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
    'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver(
    'div_up', lambda x, y: (x + y - 1) // y)
omegaconf.OmegaConf.register_new_resolver(
    'if_then_else', lambda c, x, y: x if c else y)


SA_THRESHOLDS = (3.0, 3.5, 4.0, 4.5)


def _compute_sa(mol: rdChem.Mol) -> float:
  """RDKit Contrib sascorer; lower is more synthesizable."""
  return float(sascorer.calculateScore(mol))


def _compute_qed(mol: rdChem.Mol) -> float:
  return float(QED.qed(mol))


def _evaluate_samples(
    samples: list[str],
    qm9_smiles: set[str],
) -> dict:
  """Compute Valid / Unique / Novel + per-molecule SA and QED for SMILES.

  Returns counts (not rates) plus per-molecule arrays.
  """
  valids: list[str] = []
  per_mol = {'smiles': [], 'sa': [], 'qed': [], 'is_novel': []}
  for t in samples:
    t = t.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '').strip()
    if len(t) == 0:
      continue
    try:
      mol = rdChem.MolFromSmiles(t)
    except rdkit.Chem.rdchem.KekulizeException:
      continue
    if mol is None:
      continue
    valids.append(t)
    per_mol['smiles'].append(t)
    per_mol['sa'].append(_compute_sa(mol))
    per_mol['qed'].append(_compute_qed(mol))
    per_mol['is_novel'].append(t not in qm9_smiles)

  return {
      'n_total': len(samples),
      'n_valid': len(valids),
      'n_unique': len(set(valids)),
      'per_mol': per_mol,
  }


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config: omegaconf.DictConfig) -> None:
  L.seed_everything(config.seed)

  # ── Load diffusion + classifier ─────────────────────────────────────────────
  tokenizer = dataloader.get_tokenizer(config)
  pretrained = diffusion.Diffusion.load_from_checkpoint(
      config.eval.checkpoint_path,
      tokenizer=tokenizer,
      config=config, logger=False)
  pretrained.eval()
  if torch.cuda.is_available():
    pretrained = pretrained.to('cuda')

  guidance_cfg = omegaconf.OmegaConf.select(config, 'guidance', default=None)
  if guidance_cfg is None:
    print('[sa_eval] WARNING: guidance is null — running unconditional MDLM.')
  else:
    schedule = omegaconf.OmegaConf.select(
        guidance_cfg, 'gamma_schedule', default=None)
    g_min = omegaconf.OmegaConf.select(
        guidance_cfg, 'gamma_min', default=None)
    g_max = omegaconf.OmegaConf.select(
        guidance_cfg, 'gamma_max', default=None)
    print(f'[sa_eval] guidance: method={guidance_cfg.method}, '
          f'gamma={guidance_cfg.gamma}, condition={guidance_cfg.condition}, '
          f'classifier={guidance_cfg.classifier_checkpoint_path}')
    print(f'[sa_eval] gamma_schedule={schedule}, gamma_min={g_min}, '
          f'gamma_max={g_max}')
    if schedule is not None and schedule != 'constant':
      try:
        gmin_f = float(g_min) if g_min is not None else 0.0
        gmax_f = float(g_max) if g_max is not None else float(guidance_cfg.gamma)
        import numpy as _np
        ts = [1.0, 0.75, 0.5, 0.25, 0.0]
        if schedule == 'linear_increasing':
          gs = [gmin_f + (gmax_f - gmin_f) * (1 - t) for t in ts]
        elif schedule == 'quadratic_increasing':
          gs = [gmin_f + (gmax_f - gmin_f) * (1 - t) ** 2 for t in ts]
        elif schedule == 'linear_decreasing':
          gs = [gmin_f + (gmax_f - gmin_f) * t for t in ts]
        elif schedule == 'cosine_increasing':
          gs = [gmin_f + (gmax_f - gmin_f) * 0.5 * (1 - _np.cos(_np.pi * (1 - t))) for t in ts]
        else:
          gs = []
        if gs:
          preview = ', '.join(f't={t:.2f}->{g:.2f}' for t, g in zip(ts, gs))
          print(f'[sa_eval] gamma(t) preview: {preview}')
      except Exception as _e:
        print(f'[sa_eval] gamma preview failed: {_e}')

  # ── Sample ──────────────────────────────────────────────────────────────────
  samples: list[str] = []
  ad_trajectories: list = []  # per-batch trajectories, only populated for adaptive_dual
  for _ in tqdm(range(config.sampling.num_sample_batches),
                desc='SA-eval batches', leave=False):
    sample_ids = pretrained.sample()
    samples.extend(pretrained.tokenizer.batch_decode(sample_ids))
    # If adaptive_dual ran, scrape the full per-step trajectory of this batch
    # before the next sample() overwrites it.
    traj = getattr(pretrained, '_last_adaptive_dual_traj', None)
    if traj is not None:
      ad_trajectories.append(traj)
      pretrained._last_adaptive_dual_traj = None  # avoid double-counting
  print(f'[sa_eval] generated {len(samples)} samples')
  if ad_trajectories:
    print(f'[sa_eval] captured adaptive_dual trajectory for '
          f'{len(ad_trajectories)} batches × {len(ad_trajectories[0])} steps')

  # ── Evaluate ────────────────────────────────────────────────────────────────
  print('[sa_eval] loading QM9 train for novelty check…')
  qm9 = datasets.load_dataset(
      'yairschiff/qm9', trust_remote_code=True, split='train')
  qm9_smiles_set = set(qm9['canonical_smiles'])

  result = _evaluate_samples(samples, qm9_smiles_set)
  per_mol = result['per_mol']
  n_total = result['n_total']
  n_valid = result['n_valid']
  n_unique = result['n_unique']

  sa_arr = np.array(per_mol['sa'])
  qed_arr = np.array(per_mol['qed'])
  novel_mask = np.array(per_mol['is_novel'], dtype=bool)

  # Violation rate at each tau: fraction of VALID molecules with SA > tau.
  viol_rates = {}
  for tau in SA_THRESHOLDS:
    viol_rates[tau] = float((sa_arr > tau).mean()) if n_valid > 0 else 0.0

  # CDD's "Novel" definition: valid AND not-in-train AND SA <= 3.0.
  novel_strict_mask = novel_mask & (sa_arr <= 3.0)
  n_novel_strict = int(novel_strict_mask.sum())
  qed_novel_strict = (
      float(qed_arr[novel_strict_mask].mean()) if n_novel_strict > 0 else 0.0)

  # Raw "novelty among valid" (not gating by SA).
  n_novel_unconstrained = int(novel_mask.sum())

  # ── Report ──────────────────────────────────────────────────────────────────
  print('\n=== SA-constrained MDLM D-CBG evaluation ===')
  print(f'  Total samples       : {n_total}')
  print(f'  Valid               : {n_valid:5d} ({100*n_valid/max(n_total,1):.2f}%)')
  print(f'  Unique (of valid)   : {n_unique:5d}')
  print(f'  Novel (of valid)    : {n_novel_unconstrained:5d} '
        f'({100*n_novel_unconstrained/max(n_valid,1):.2f}%)')
  print(f'  Novel & SA<=3.0     : {n_novel_strict:5d}  <-- CDD "Novel" column')
  print(f'  QED on novel-strict : {qed_novel_strict:.3f}  <-- CDD "QED" column')
  print(f'  Violation rates (frac valid with SA > tau):')
  for tau in SA_THRESHOLDS:
    print(f'    tau={tau}: {100*viol_rates[tau]:.2f}%')

  print('\nReference (CDD paper, MDLM_D-CBG gamma=3):')
  print('  Valid=436, Novel=14, QED=0.37, '
        'Viol: 50.5/48.6/46.1/44.7% @ tau=3.0/3.5/4.0/4.5')

  # ── Save ────────────────────────────────────────────────────────────────────
  samples_path = config.eval.generated_samples_path
  csv_path = omegaconf.OmegaConf.select(
      config, 'eval.results_csv_path', default=None)

  if samples_path:
    with open(samples_path, 'w') as f:
      json.dump({
          'samples': samples,
          'per_mol': per_mol,
          'n_total': n_total,
          'n_valid': n_valid,
          'n_unique': n_unique,
          'n_novel_unconstrained': n_novel_unconstrained,
          'n_novel_strict_sa3.0': n_novel_strict,
          'qed_novel_strict': qed_novel_strict,
          'viol_rates': {str(k): v for k, v in viol_rates.items()},
      }, f, indent=2)
    print(f'[sa_eval] wrote samples + stats to {samples_path}')

  if csv_path:
    row = {
        'method': 'mdlm_dcbg',
        'gamma': omegaconf.OmegaConf.select(
            config, 'guidance.gamma', default=None),
        'condition': omegaconf.OmegaConf.select(
            config, 'guidance.condition', default=None),
        'n_total': n_total,
        'valid': n_valid,
        'unique': n_unique,
        'novel_unconstrained': n_novel_unconstrained,
        'novel_strict_sa3.0': n_novel_strict,
        'qed_novel_strict': qed_novel_strict,
        **{f'viol_tau_{tau}': viol_rates[tau] for tau in SA_THRESHOLDS},
    }
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    print(f'[sa_eval] wrote summary row to {csv_path}')

  # ── Adaptive-dual trajectory dump (only for adaptive_dual runs) ────────────
  if ad_trajectories and samples_path:
    traj_path = samples_path.replace('_samples.json', '_traj.json')
    if traj_path == samples_path:  # fallback if filename doesn't follow convention
      traj_path = samples_path + '.traj.json'
    traj_payload = {
        'config': {
            'schedule': omegaconf.OmegaConf.select(
                config, 'guidance.gamma_schedule', default=None),
            'gamma_C': omegaconf.OmegaConf.select(
                config, 'guidance.gamma_C', default=None),
            'gamma_rho': omegaconf.OmegaConf.select(
                config, 'guidance.gamma_rho', default=None),
            'gamma_lambda_init': omegaconf.OmegaConf.select(
                config, 'guidance.gamma_lambda_init', default=None),
            'gamma_lambda_max': omegaconf.OmegaConf.select(
                config, 'guidance.gamma_lambda_max', default=None),
            'sampling_steps': config.sampling.steps,
            'batch_size': config.loader.eval_global_batch_size,
            'num_batches': config.sampling.num_sample_batches,
        },
        # Shape: (n_batches, n_steps) list of dicts with per-sample λ + log_p_y
        # arrays of length B (= batch size). Plus per-step scalar aggregates.
        'trajectories': ad_trajectories,
    }
    with open(traj_path, 'w') as f:
      json.dump(traj_payload, f)
    print(f'[sa_eval] wrote adaptive_dual per-step trajectory to {traj_path}')


if __name__ == '__main__':
  main()
