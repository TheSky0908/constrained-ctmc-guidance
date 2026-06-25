"""Summarize novelty D-CBG rollout/timedep-classifier results into markdown tables.

Reads the per-gamma result CSVs written by sweep_gamma_rollout.sh (and, for
side-by-side comparison, the forward-classifier CSVs from sweep_gamma.sh) and
prints constant-gamma markdown tables. Columns mirror novelty_dcbg_eval.md:
  Valid | Unique | novel_rate | Viol% | Valid&Novel | QED(novel)

Usage:
  python experiments/molecule_novelty/analysis/summarize_rollout.py [STEPS_SUFFIX]
    STEPS_SUFFIX: "" for steps=128 (default), or e.g. "_steps256".
"""
import csv
import os
import sys

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')


def _read(path):
  if not os.path.isfile(path):
    return None
  with open(path) as f:
    rows = list(csv.DictReader(f))
  return rows[-1] if rows else None


def _fmt_row(tag, r):
  if r is None:
    return f'| {tag} | _pending_ | | | | | |'
  valid = int(r['valid'])
  unique = int(r['unique'])
  vn = int(r['valid_novel'])
  nr = float(r['novel_rate']) if r['novel_rate'] else 0.0
  viol = float(r['viol']) if r['viol'] else 0.0
  qed = float(r['qed_novel']) if r['qed_novel'] else float('nan')
  qed_s = f'{qed:.3f}' if qed == qed else '—'
  return (f'| {tag} | {valid} | {unique} | {nr*100:.1f}% | '
          f'{viol*100:.2f}% | {vn} | {qed_s} |')


def table(tag_suffix, label):
  print(f'\n### {label}\n')
  print('| Method | Valid | Unique | novel_rate | Viol % | **Valid & Novel** | QED |')
  print('| :----- | ----: | -----: | ---------: | -----: | ----------------: | --: |')
  # gamma=0 = no-guidance
  ng = _read(os.path.join(RESULTS, f'noguidance{tag_suffix}_results.csv'))
  print(_fmt_row('γ = 0 (no-guidance)', ng))
  for g in range(1, 11):
    r = _read(os.path.join(RESULTS, f'dcbg_gamma{g}{tag_suffix}_results.csv'))
    print(_fmt_row(f'CBG γ = {g}', r))


def main():
  steps_suf = sys.argv[1] if len(sys.argv) > 1 else ''
  print(f'# rollout/timedep constant-γ summary (suffix={steps_suf!r})')
  table(f'_rollout{steps_suf}', 'ROLLOUT classifier (eq.3, time-dependent)')
  table(f'{steps_suf}', 'FORWARD classifier (original, time-independent) — for comparison')


if __name__ == '__main__':
  main()
