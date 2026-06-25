"""Plot the constant-γ D-CBG novelty trade-off: ROLLOUT (eq.3, time-dependent)
classifier vs the original FORWARD (time-independent) classifier, steps=128, N=500.

Two panels:
  (A) Viol% (x) vs Valid&Novel (y) — the headline constraint/quality frontier,
      each point a γ (annotated). Up-and-left = better.
  (B) γ (x) vs Viol% and Valid (twin y) — shows the rollout classifier's
      compressed γ-axis (stronger guide) and faster validity collapse.

Reads results/{noguidance,dcbg_gamma{1..10}}{,_rollout}_results.csv.
Saves figures/rollout_vs_forward_steps128.png.
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, '..', 'results')
FIGDIR = os.path.join(HERE, '..', 'figures')


def _read(path):
  if not os.path.isfile(path):
    return None
  with open(path) as f:
    rows = list(csv.DictReader(f))
  return rows[-1] if rows else None


def series(tag_suffix):
  """Return dict gamma -> (valid, valid_novel, viol_pct) for gamma 0..10."""
  out = {}
  for g in range(0, 11):
    if g == 0:
      r = _read(os.path.join(RESULTS, f'noguidance{tag_suffix}_results.csv'))
    else:
      r = _read(os.path.join(RESULTS, f'dcbg_gamma{g}{tag_suffix}_results.csv'))
    if r is None:
      continue
    valid = int(r['valid'])
    vn = int(r['valid_novel'])
    viol = float(r['viol']) * 100 if r['viol'] and valid > 0 else None
    out[g] = (valid, vn, viol)
  return out


def main():
  roll = series('_rollout')
  fwd = series('')

  os.makedirs(FIGDIR, exist_ok=True)
  fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

  # ---- Panel A: Viol% vs Valid&Novel (only gammas with valid>0 & viol defined)
  for data, label, color, marker in [
      (roll, 'rollout (eq.3, time-dependent)', 'tab:red', 'o'),
      (fwd, 'forward (original, time-independent)', 'tab:blue', 's')]:
    xs, ys, gs = [], [], []
    for g in sorted(data):
      valid, vn, viol = data[g]
      if viol is None or valid == 0:
        continue
      xs.append(viol); ys.append(vn); gs.append(g)
    axA.plot(xs, ys, marker=marker, color=color, label=label, alpha=0.85, lw=2)
    for x, y, g in zip(xs, ys, gs):
      axA.annotate(f'γ{g}', (x, y), textcoords='offset points',
                   xytext=(5, 4), fontsize=8, color=color)
  axA.set_xlabel('Viol %  (= 1 − novel_rate, lower better)')
  axA.set_ylabel('Valid & Novel  (count, higher better)')
  axA.set_title('(A) Constraint–quality frontier (steps=128, N=500)')
  axA.grid(True, alpha=0.3)
  axA.legend(fontsize=9, loc='upper right')

  # ---- Panel B: gamma vs Viol% (left y) and Valid (right y)
  axB2 = axB.twinx()
  for data, label, color in [
      (roll, 'rollout (td)', 'tab:red'),
      (fwd, 'forward (ti)', 'tab:blue')]:
    gv = sorted(data)
    viol_x = [g for g in gv if data[g][2] is not None]
    viol_y = [data[g][2] for g in viol_x]
    valid_y = [data[g][0] for g in gv]
    axB.plot(viol_x, viol_y, marker='o', color=color, lw=2,
             label=f'{label} Viol%')
    axB2.plot(gv, valid_y, marker='x', linestyle='--', color=color, alpha=0.6,
              label=f'{label} Valid')
  axB.set_xlabel('γ (constant guidance strength)')
  axB.set_ylabel('Viol %  (solid ●, lower better)')
  axB2.set_ylabel('Valid count  (dashed ✕)')
  axB.set_title('(B) γ-axis: stronger guide + faster validity collapse')
  axB.grid(True, alpha=0.3)
  # Merge both axes' handles so the dashed Valid series are labelled too.
  h1, l1 = axB.get_legend_handles_labels()
  h2, l2 = axB2.get_legend_handles_labels()
  axB.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right')

  fig.suptitle('Novelty D-CBG: rollout (time-dependent) vs forward classifier — '
               'constant γ, steps=128, N=500, seed=1', fontsize=11)
  fig.tight_layout(rect=[0, 0, 1, 0.96])
  out = os.path.join(FIGDIR, 'rollout_vs_forward_steps128.png')
  fig.savefig(out, dpi=150)
  print(f'saved -> {out}')


if __name__ == '__main__':
  main()
