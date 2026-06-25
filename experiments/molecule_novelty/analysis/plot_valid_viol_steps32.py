"""Valid–Viol trade-off at steps=32 (N=500) for three novelty D-CBG methods,
all using the ROLLOUT (eq.3, time-dependent) classifier:

  1. constant γ          — dcbg_gamma{0..7}_rollout_steps32_results.csv
  2. adaptive-dual sf+td  — adual_sftd_search_steps32.csv   (sample_first, Algm 1)
  3. adaptive-dual IL+td  — adual_iltd_search_steps32.csv   (inner-loop, Algm 2)

x = Viol % (= 1 − novel_rate, lower better); y = Valid count (higher better).
Each config is a point; the Pareto frontier (max Valid / min Viol, up-and-left)
is drawn as a solid line per method. Saves figures/valid_viol_steps32.png.
Mirrors plot_valid_viol_steps64.py.
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, '..', 'results')
FIGDIR = os.path.join(HERE, '..', 'figures')


def _read_last(path):
  if not os.path.isfile(path):
    return None
  with open(path) as f:
    rows = list(csv.DictReader(f))
  return rows[-1] if rows else None


def constant_gamma_points():
  """(viol%, valid, label) for γ=0..7, only valid>0 & viol defined."""
  pts = []
  for g in range(0, 8):
    fn = ('noguidance_rollout_steps32_results.csv' if g == 0
          else f'dcbg_gamma{g}_rollout_steps32_results.csv')
    r = _read_last(os.path.join(RESULTS, fn))
    if r is None:
      continue
    valid = int(r['valid'])
    if valid == 0 or not r['viol']:
      continue
    pts.append((float(r['viol']) * 100, valid, f'γ{g}'))
  return pts


def search_points(fname):
  """(viol%, valid, run_name) for every row of a *_search_steps32.csv."""
  path = os.path.join(RESULTS, fname)
  if not os.path.isfile(path):
    return []
  pts = []
  with open(path) as f:
    for r in csv.DictReader(f):
      if not r.get('valid') or r['valid'] == 'FAIL':
        continue
      valid = int(r['valid'])
      if valid == 0 or not r['viol']:
        continue
      pts.append((float(r['viol']) * 100, valid, r['run_name']))
  return pts


def pareto_frontier(points):
  """Keep points not dominated on (low viol, high valid). Sorted by viol asc."""
  pts = sorted(points, key=lambda p: (p[0], -p[1]))
  front, best_valid = [], -1
  for viol, valid, lab in pts:
    if valid > best_valid:
      front.append((viol, valid, lab))
      best_valid = valid
  return front


def draw(methods, out, viol_max=None):
  fig, ax = plt.subplots(figsize=(8.5, 6.5))
  for label, pts, color, marker, trim_top in methods:
    if not pts:
      print(f'WARN no points for {label}')
      continue
    front = pareto_frontier(pts)
    if viol_max is not None:
      front = [p for p in front if p[0] <= viol_max]
    if trim_top:
      front = sorted(front, key=lambda p: p[0])[:-trim_top]
    fx = [p[0] for p in front]
    fy = [p[1] for p in front]
    ax.plot(fx, fy, color=color, lw=2, marker=marker, ms=7,
            label=label, zorder=4)
    if label == 'constant γ':
      for viol, valid, lab in front:
        ax.annotate(lab, (viol, valid), textcoords='offset points',
                    xytext=(5, 4), fontsize=8, color=color)

  ax.set_xlabel('Viol %  (← lower better)')
  ax.set_ylabel('Valid  (count / 500, higher better ↑)')
  title = ('Novelty D-CBG Valid–Viol trade-off (steps=32, N=500)\n'
           'rollout time-dependent classifier; lines = Pareto frontier')
  if viol_max is not None:
    title += f'  —  Viol < {viol_max:g}%'
    ax.set_xlim(-1, viol_max)
  ax.set_title(title)
  ax.grid(True, alpha=0.3)
  ax.legend(fontsize=9, loc='lower right')

  fig.tight_layout()
  fig.savefig(out, dpi=150)
  plt.close(fig)
  print(f'saved -> {out}')


def main():
  methods = [
      ('constant γ', constant_gamma_points(), 'tab:blue', 'o', 0),
      ('adaptive-dual sf+td', search_points('adual_sftd_search_steps32.csv'),
       'tab:green', '^', 0),
      ('adaptive-dual IL+td', search_points('adual_iltd_search_steps32.csv'),
       'tab:red', 's', 0),
  ]

  os.makedirs(FIGDIR, exist_ok=True)
  draw(methods, os.path.join(FIGDIR, 'valid_viol_steps32.png'))
  draw(methods, os.path.join(FIGDIR, 'valid_viol_steps32_viol20.png'),
       viol_max=20)

  for label, pts, _, _, _ in methods:
    if not pts:
      continue
    front = pareto_frontier(pts)
    lo = min(front, key=lambda p: p[0])
    hi = max(front, key=lambda p: p[1])
    print(f'{label:22s} | low-Viol: Viol={lo[0]:5.2f}% Valid={lo[1]:3d} ({lo[2]})'
          f' | max-Valid: Viol={hi[0]:5.2f}% Valid={hi[1]:3d}')


if __name__ == '__main__':
  main()
