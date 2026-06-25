"""Effective QED = mean over ALL N samples of per-molecule QED, with invalid
molecules assigned QED=0 (i.e. sum of valid molecules' QED / n_total).

Contrast with the table's existing 'QED' column = mean QED on the novel set only.
Effective QED rewards a method for producing many valid, drug-like molecules and
penalises validity collapse — a config that keeps Viol low by emitting only a
handful of valid molecules scores poorly here.

Reads the per-run *_samples.json (per_mol.qed holds QED for every valid molecule;
invalid ones are never appended, so they count as 0). Prints (valid, viol%,
qed_novel, eff_qed) keyed by run so table rows can be matched by (valid, viol).
Run: python experiments/molecule_novelty/analysis/compute_effective_qed.py
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, '..', 'results')


def eff_qed(path):
  d = json.load(open(path))
  q = np.array(d['per_mol']['qed'])
  return q.sum() / d['n_total'], d['n_valid'], d['viol'] * 100, d['qed_novel']


def report(name, path):
  if not os.path.isfile(path):
    print(f'  [missing] {name}')
    return
  e, v, vi, qn = eff_qed(path)
  print(f'  {name:60s} valid={v:3d} viol={vi:5.2f}% '
        f'qed_novel={qn:.3f} eff_qed={e:.3f}')


def main():
  print('== constant gamma (rollout, steps=64) ==')
  for g in range(0, 6):
    fn = ('noguidance_rollout_steps64_samples.json' if g == 0
          else f'dcbg_gamma{g}_rollout_steps64_samples.json')
    report(f'gamma{g}', os.path.join(RESULTS, fn))

  for tag, sub in [('sf+td', 'adual_sftd_search'),
                   ('IL+td', 'adual_iltd_search')]:
    master = os.path.join(RESULTS, f'adual_{tag.split("+")[0]}td_search_steps64.csv')
    print(f'\n== adaptive-dual {tag} (steps=64) ==')
    if not os.path.isfile(master):
      print('  [missing master csv]')
      continue
    for r in csv.DictReader(open(master)):
      report(r['run_name'],
             os.path.join(RESULTS, sub, r['run_name'] + '_samples.json'))


if __name__ == '__main__':
  main()
