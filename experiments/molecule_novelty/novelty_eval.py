"""Novelty-constrained QM9 evaluation for MDLM + D-CBG (adaptive-dual gamma).

Mirror of sa_eval.py for the molecule *novelty* task (CDD paper, NeurIPS 2025,
arXiv:2503.09790, Fig 4 RIGHT table). The guidance classifier predicts
p(novel | x_t) (class 1 = novel); the adaptive-dual schedule adaptively presses
the novelty constraint via the dual variable lambda. Headline KPI = number of
*valid & novel* molecules (maximize), with QED reported on the novel set.

CDD paper "No Guidance" row (92M, valid&novel / QED):
  AR 11 / 0.41 ; MDLM 271 / 0.45 ; UDLM 345 / 0.46 ; CDD 511 / 0.45

Run:
  CUDA_VISIBLE_DEVICES=0 python experiments/molecule_novelty/novelty_eval.py \
    data=qm9 model=small backbone=dit model.length=32 \
    parameterization=subs diffusion=absorbing_state \
    time_conditioning=False T=0 \
    eval.checkpoint_path=outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt \
    guidance=cbg guidance.method=cbg guidance.condition=1 \
    guidance.gamma_schedule=adaptive_dual \
    guidance.classifier_checkpoint_path=<novelty_classifier.ckpt> \
    sampling.steps=128 sampling.num_sample_batches=4 \
    loader.eval_global_batch_size=256
"""
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
from rdkit import Chem as rdChem
from rdkit.Chem import QED
from tqdm.auto import tqdm

import dataloader
import diffusion

rdkit.rdBase.DisableLog('rdApp.error')

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
    'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver(
    'div_up', lambda x, y: (x + y - 1) // y)
omegaconf.OmegaConf.register_new_resolver(
    'if_then_else', lambda c, x, y: x if c else y)


def _compute_qed(mol: rdChem.Mol) -> float:
  return float(QED.qed(mol))


def _evaluate_samples(samples: list[str], qm9_smiles: set[str]) -> dict:
  """Compute Valid / Unique / Novel + per-molecule QED. Counts, not rates."""
  valids: list[str] = []
  per_mol = {'smiles': [], 'qed': [], 'is_novel': []}
  for t in samples:
    t = t.replace('<bos>', '').replace('<eos>', '').replace(
      '<pad>', '').strip()
    if len(t) == 0:
      continue
    try:
      mol = rdChem.MolFromSmiles(t)
    except rdkit.Chem.rdchem.KekulizeException:
      continue
    if mol is None:
      continue
    # Canonicalize so novelty/uniqueness match QM9's canonical_smiles space.
    canon = rdChem.MolToSmiles(mol)
    valids.append(canon)
    per_mol['smiles'].append(canon)
    per_mol['qed'].append(_compute_qed(mol))
    per_mol['is_novel'].append(canon not in qm9_smiles)
  return {
      'n_total': len(samples),
      'n_valid': len(valids),
      'n_unique': len(set(valids)),
      'valids': valids,
      'per_mol': per_mol,
  }


@hydra.main(version_base=None, config_path='../../configs',
            config_name='config')
def main(config: omegaconf.DictConfig) -> None:
  L.seed_everything(config.seed)

  # ── Load diffusion + classifier ─────────────────────────────────────────────
  tokenizer = dataloader.get_tokenizer(config)
  pretrained = diffusion.Diffusion.load_from_checkpoint(
      config.eval.checkpoint_path, tokenizer=tokenizer,
      config=config, logger=False)
  pretrained.eval()
  if torch.cuda.is_available():
    pretrained = pretrained.to('cuda')

  guidance_cfg = omegaconf.OmegaConf.select(config, 'guidance', default=None)
  if guidance_cfg is None:
    print('[novelty_eval] WARNING: guidance is null — unconditional MDLM.')
  else:
    schedule = omegaconf.OmegaConf.select(
        guidance_cfg, 'gamma_schedule', default=None)
    print(f'[novelty_eval] guidance: method={guidance_cfg.method}, '
          f'gamma={guidance_cfg.gamma}, condition={guidance_cfg.condition}, '
          f'classifier={guidance_cfg.classifier_checkpoint_path}')
    print(f'[novelty_eval] gamma_schedule={schedule}, '
          f'C={omegaconf.OmegaConf.select(guidance_cfg, "gamma_C", default=None)}, '
          f'rho={omegaconf.OmegaConf.select(guidance_cfg, "gamma_rho", default=None)}, '
          f'lambda_max={omegaconf.OmegaConf.select(guidance_cfg, "gamma_lambda_max", default=None)}')

  # ── Sample ──────────────────────────────────────────────────────────────────
  samples: list[str] = []
  ad_trajectories: list = []  # per-batch, only populated for adaptive_dual
  for _ in tqdm(range(config.sampling.num_sample_batches),
                desc='novelty-eval batches', leave=False):
    sample_ids = pretrained.sample()
    samples.extend(pretrained.tokenizer.batch_decode(sample_ids))
    traj = getattr(pretrained, '_last_adaptive_dual_traj', None)
    if traj is not None:
      ad_trajectories.append(traj)
      pretrained._last_adaptive_dual_traj = None
  print(f'[novelty_eval] generated {len(samples)} samples')
  if ad_trajectories:
    print(f'[novelty_eval] captured adaptive_dual trajectory for '
          f'{len(ad_trajectories)} batches × {len(ad_trajectories[0])} steps')

  # ── Evaluate ────────────────────────────────────────────────────────────────
  print('[novelty_eval] loading QM9 train for novelty check…')
  qm9 = datasets.load_dataset(
      'yairschiff/qm9', trust_remote_code=True, split='train')
  qm9_smiles_set = set(qm9['canonical_smiles'])

  result = _evaluate_samples(samples, qm9_smiles_set)
  per_mol = result['per_mol']
  n_total = result['n_total']
  n_valid = result['n_valid']
  n_unique = result['n_unique']

  qed_arr = np.array(per_mol['qed'])
  novel_mask = np.array(per_mol['is_novel'], dtype=bool)

  # Headline: valid & novel. Paper counts distinct novel molecules.
  n_valid_novel = int(novel_mask.sum())
  novel_smiles = [s for s, nv in zip(per_mol['smiles'], novel_mask) if nv]
  n_unique_novel = len(set(novel_smiles))
  qed_novel = float(qed_arr[novel_mask].mean()) if n_valid_novel > 0 else 0.0
  novel_rate = n_valid_novel / max(n_valid, 1)
  # CDD "Viol (%)" = fraction of VALID generations that are NOT novel
  # (i.e. already in QM9 train). Equivalently 1 - novel_rate.
  viol = 1.0 - novel_rate if n_valid > 0 else 0.0

  # ── Report ──────────────────────────────────────────────────────────────────
  print('\n=== Novelty-constrained MDLM D-CBG (adaptive-dual) evaluation ===')
  print(f'  Total samples         : {n_total}')
  print(f'  Valid                 : {n_valid:5d} '
        f'({100*n_valid/max(n_total,1):.2f}%)')
  print(f'  Unique (of valid)     : {n_unique:5d}')
  print(f'  Valid & Novel         : {n_valid_novel:5d}  '
        f'<-- CDD "Valid & Novel" column  (rate {100*novel_rate:.2f}%)')
  print(f'  Unique Valid & Novel  : {n_unique_novel:5d}')
  print(f'  Viol (not-novel/valid): {100*viol:.2f}%  <-- CDD "Viol (%)" column')
  print(f'  QED on novel          : {qed_novel:.3f}  <-- CDD "QED" column')
  print('\nReference (CDD paper, No Guidance, valid&novel / QED):')
  print('  AR 11/0.41 | MDLM 271/0.45 | UDLM 345/0.46 | CDD 511/0.45')

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
          'n_valid_novel': n_valid_novel,
          'n_unique_novel': n_unique_novel,
          'qed_novel': qed_novel,
          'novel_rate': novel_rate,
          'viol': viol,
      }, f, indent=2)
    print(f'[novelty_eval] wrote samples + stats to {samples_path}')

  if csv_path:
    row = {
        'method': 'mdlm_dcbg_novelty',
        'gamma': omegaconf.OmegaConf.select(
            config, 'guidance.gamma', default=None),
        'gamma_schedule': omegaconf.OmegaConf.select(
            config, 'guidance.gamma_schedule', default=None),
        'gamma_C': omegaconf.OmegaConf.select(
            config, 'guidance.gamma_C', default=None),
        'gamma_rho': omegaconf.OmegaConf.select(
            config, 'guidance.gamma_rho', default=None),
        'gamma_lambda_max': omegaconf.OmegaConf.select(
            config, 'guidance.gamma_lambda_max', default=None),
        'condition': omegaconf.OmegaConf.select(
            config, 'guidance.condition', default=None),
        'n_total': n_total,
        'valid': n_valid,
        'unique': n_unique,
        'valid_novel': n_valid_novel,
        'unique_novel': n_unique_novel,
        'qed_novel': qed_novel,
        'novel_rate': novel_rate,
        'viol': viol,
    }
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    print(f'[novelty_eval] wrote summary row to {csv_path}')

  # ── Adaptive-dual trajectory dump ──────────────────────────────────────────
  if ad_trajectories and samples_path:
    traj_path = samples_path.replace('_samples.json', '_traj.json')
    if traj_path == samples_path:
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
        'trajectories': ad_trajectories,
    }
    with open(traj_path, 'w') as f:
      json.dump(traj_payload, f)
    print(f'[novelty_eval] wrote adaptive_dual trajectory to {traj_path}')


if __name__ == '__main__':
  main()
