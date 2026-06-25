"""Label held-out Jigsaw text with the GPT-Neo surrogate's continuous toxicity
score, to train the per-tau D-CBG / adaptive-dual guidance classifiers.

Why held-out: the surrogate is overconfident on its own training split (scores
collapse to ~0/1), which would make the tau in {0.25,0.50,0.75} thresholds nearly
identical. We therefore score the Jigsaw **test** split (never seen by the
surrogate), where scores span [0,1] and the three thresholds are genuinely
distinct — analogous to RDKit giving a continuous SA score in the molecule task.

Output: a HF DatasetDict {train, validation} with columns `comment_text` +
`toxicity_score`, saved to disk and loaded by the `jigsaw_scored` dataloader
branch (binarized as class 1 = `toxicity_score <= tau`).

Run:
  python label_jigsaw_with_surrogate.py \
    --surrogate_dir outputs/owt/toxic_surrogate \
    --num 60000 --out_dir .data_cache/jigsaw_surrogate_scored
"""
import argparse

import datasets
import numpy as np

import toxicity_scorer


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--surrogate_dir', default='outputs/owt/toxic_surrogate')
  ap.add_argument('--out_dir', default='.data_cache/jigsaw_surrogate_scored')
  ap.add_argument('--split', default='test',
                  help='Jigsaw split to score (held out from the surrogate).')
  ap.add_argument('--num', type=int, default=60000,
                  help='Subsample this many comments (<=0 uses all).')
  ap.add_argument('--batch_size', type=int, default=64)
  ap.add_argument('--max_length', type=int, default=256)
  ap.add_argument('--val_frac', type=float, default=0.05)
  ap.add_argument('--seed', type=int, default=42)
  args = ap.parse_args()

  ds = datasets.load_dataset(
      'Arsive/toxicity_classification_jigsaw', trust_remote_code=True,
      split=args.split)
  texts = [t for t in ds['comment_text'] if t and t.strip()]
  rng = np.random.default_rng(args.seed)
  if args.num > 0 and args.num < len(texts):
    sel = rng.permutation(len(texts))[:args.num]
    texts = [texts[i] for i in sel]
  print(f'[label] scoring {len(texts)} held-out comments with surrogate '
        f'{args.surrogate_dir}…', flush=True)

  scorer = toxicity_scorer.ToxicityScorer(
      args.surrogate_dir, batch_size=args.batch_size,
      max_length=args.max_length)
  scores = scorer.score(texts)

  # Distribution sanity check — confirm scores are spread across [0,1], not
  # collapsed to the extremes (which would make the tau thresholds identical).
  edges = [0, .1, .25, .5, .75, .9, 1.01]
  hist, _ = np.histogram(scores, bins=edges)
  print('[label] score histogram (bins '
        '[0,.1,.25,.5,.75,.9,1]):', hist.tolist())
  for tau in (0.25, 0.50, 0.75):
    frac = float((scores <= tau).mean())
    print(f'[label]   frac(score<=tau={tau}) = {frac:.3f}  '
          f'(positive class for that classifier)')

  out = datasets.Dataset.from_dict(
      {'comment_text': texts, 'toxicity_score': scores.tolist()})
  split = out.train_test_split(test_size=args.val_frac, seed=args.seed)
  dd = datasets.DatasetDict({'train': split['train'],
                             'validation': split['test']})
  dd.save_to_disk(args.out_dir)
  print(f'[label] saved scored dataset to {args.out_dir} '
        f"(train={len(dd['train'])}, val={len(dd['validation'])})", flush=True)


if __name__ == '__main__':
  main()
