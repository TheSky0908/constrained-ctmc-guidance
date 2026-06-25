"""Summarize the gamma-schedule experiment results vs the constant-gamma baseline.

Reads:
  outputs/qm9/mdlm_no-guidance/mdlm_dcbg_sa_n500_<schedule>_gmin<gmin>_gmax<gmax>_trainTau3.0_samples.json
  outputs/qm9/mdlm_no-guidance/mdlm_dcbg_sa_gamma3_trainTau3.0_samples.json  (baseline)

Prints a markdown table of Valid / Unique / Novel-strict / Viol@tau / QED.
"""
import glob
import json
import os
import re

ROOT = '/local/scratch/zhiheng/guidance/outputs/qm9/mdlm_no-guidance'

BASELINE = os.path.join(ROOT, 'mdlm_dcbg_sa_gamma3_trainTau3.0_samples.json')

PATTERN = re.compile(
    r'mdlm_dcbg_sa_n500_([a-z_]+)_gmin([\d.]+)_gmax([\d.]+)_trainTau3\.0_samples\.json$'
)
ADUAL_PATTERN = re.compile(
    r'mdlm_dcbg_sa_n500_C([\d.]+)_rho([\d.]+)_l0([\d.]+)_lmax([\d.]+)_trainTau3\.0_samples\.json$'
)


def load_row(path: str, label: str) -> dict:
  with open(path) as f:
    d = json.load(f)
  return {
      'label': label,
      'n_total': d['n_total'],
      'n_valid': d['n_valid'],
      'n_unique': d['n_unique'],
      'n_novel_strict': d['n_novel_strict_sa3.0'],
      'qed_novel_strict': d['qed_novel_strict'],
      'viol_3.0': d['viol_rates']['3.0'],
      'viol_3.5': d['viol_rates']['3.5'],
      'viol_4.0': d['viol_rates']['4.0'],
      'viol_4.5': d['viol_rates']['4.5'],
  }


def main():
  rows = []
  if os.path.exists(BASELINE):
    rows.append(load_row(BASELINE, 'constant γ=3 (N=1000, prior)'))

  for path in sorted(glob.glob(os.path.join(ROOT, 'mdlm_dcbg_sa_n500_*_samples.json'))):
    fn = os.path.basename(path)
    m_static = PATTERN.search(fn)
    m_adual = ADUAL_PATTERN.search(fn)
    if m_static:
      sched, gmin, gmax = m_static.group(1), m_static.group(2), m_static.group(3)
      label = f'{sched} [{gmin}→{gmax}]'
    elif m_adual:
      C, rho, l0, lmax = m_adual.group(1), m_adual.group(2), m_adual.group(3), m_adual.group(4)
      label = f'adaptive_dual [C={C} ρ={rho}]'
    else:
      continue
    rows.append(load_row(path, label))

  if not rows:
    print('No matching files found.')
    return

  print('\n## γ-schedule sweep — train τ = eval τ = 3.0\n')
  print('| Setup | N | Valid | Unique | Novel & SA≤3.0 | QED-novel-strict | Viol@3.0 | Viol@3.5 | Viol@4.0 | Viol@4.5 |')
  print('| :---- | -: | -----: | -----: | -------------: | ---------------: | -------: | -------: | -------: | -------: |')
  for r in rows:
    print(
      f"| {r['label']} | {r['n_total']} | "
      f"{r['n_valid']} ({100*r['n_valid']/max(r['n_total'],1):.1f}%) | "
      f"{r['n_unique']} | {r['n_novel_strict']} | {r['qed_novel_strict']:.3f} | "
      f"{100*r['viol_3.0']:.2f}% | {100*r['viol_3.5']:.2f}% | "
      f"{100*r['viol_4.0']:.2f}% | {100*r['viol_4.5']:.2f}% |"
    )

  # Identify best on each metric
  print('\n### Best by metric (lower viol = better, higher novel = better)\n')
  for metric, lower_better in [
      ('viol_3.0', True),
      ('viol_3.5', True),
      ('n_novel_strict', False),
      ('qed_novel_strict', False),
  ]:
    if lower_better:
      best = min(rows, key=lambda r: r[metric])
    else:
      best = max(rows, key=lambda r: r[metric])
    val = best[metric]
    if 'viol' in metric:
      val = f'{100*val:.2f}%'
    print(f'- **{metric}** (best: {"min" if lower_better else "max"}): `{best["label"]}` → {val}')


if __name__ == '__main__':
  main()
