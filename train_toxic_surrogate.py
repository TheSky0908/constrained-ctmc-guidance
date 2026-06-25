"""Fine-tune a GPT-Neo sequence classifier on Jigsaw as the external toxicity
violation scorer (CDD-style surrogate).

The six Jigsaw toxicity sub-labels are consolidated into a single binary target
(label 1 = toxic). The model outputs a continuous P(toxic) consumed at eval time
by ``toxicity_scorer.ToxicityScorer`` / ``tox_eval.py`` (and used to label the
per-tau guidance classifiers). To match the CDD paper, default model is
``EleutherAI/gpt-neo-1.3B``.

Self-contained PyTorch loop (no HF Trainer / accelerate). Evaluates on the FULL
validation set every ``--eval_steps`` and keeps the BEST checkpoint by val F1
(self-contained early stopping); reports accuracy / precision / recall / F1 /
ROC-AUC.

Run:
  python train_toxic_surrogate.py --model EleutherAI/gpt-neo-1.3B \
    --output_dir outputs/owt/toxic_surrogate --epochs 3 --batch_size 16
"""
import argparse
import glob
import os

import datasets
import numpy as np
import torch
import transformers
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

_SUB_LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
               'identity_hate']


def _binarize(ex):
  ex['label'] = int(any(int(ex[c]) > 0 for c in _SUB_LABELS))
  return ex


@torch.no_grad()
def _evaluate(model, loader, device):
  model.eval()
  probs, labels = [], []
  for batch in loader:
    labels.append(batch.pop('labels').numpy())
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.autocast('cuda', dtype=torch.bfloat16):
      logits = model(**batch).logits.float()
    probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
  probs = np.concatenate(probs)
  labels = np.concatenate(labels)
  preds = (probs >= 0.5).astype(int)
  acc = float((preds == labels).mean())
  tp = int(((preds == 1) & (labels == 1)).sum())
  fp = int(((preds == 1) & (labels == 0)).sum())
  fn = int(((preds == 0) & (labels == 1)).sum())
  prec = tp / (tp + fp) if tp + fp else 0.0
  rec = tp / (tp + fn) if tp + fn else 0.0
  f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
  try:
    auc = float(roc_auc_score(labels, probs))
  except ValueError:
    auc = float('nan')
  model.train()
  return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
          'auc': auc}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--model', default='EleutherAI/gpt-neo-1.3B')
  ap.add_argument('--output_dir', default='outputs/owt/toxic_surrogate')
  ap.add_argument('--epochs', type=float, default=3.0)
  ap.add_argument('--max_steps', type=int, default=-1,
                  help='If >0, overrides --epochs (useful for smoke tests).')
  ap.add_argument('--batch_size', type=int, default=16)
  ap.add_argument('--grad_accum', type=int, default=1)
  ap.add_argument('--lr', type=float, default=1e-5)
  ap.add_argument('--max_length', type=int, default=256)
  ap.add_argument('--eval_steps', type=int, default=-1,
                  help='Eval on full val set every N optimizer steps. '
                       'Default: ~6 evals across the whole run.')
  ap.add_argument('--seed', type=int, default=1)
  args = ap.parse_args()

  torch.manual_seed(args.seed)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  raw = datasets.load_dataset(
      'Arsive/toxicity_classification_jigsaw', trust_remote_code=True)
  ds = datasets.DatasetDict({
      'train': raw['train'], 'validation': raw['validation']}).map(_binarize)

  tok = transformers.AutoTokenizer.from_pretrained(args.model)
  if tok.pad_token is None:
    tok.pad_token = tok.eos_token

  def _tok(ex):
    out = tok(ex['comment_text'], truncation=True, max_length=args.max_length)
    out['labels'] = ex['label']
    return out

  ds = ds.map(_tok, remove_columns=list(ds['train'].column_names))
  ds.set_format('torch')
  collator = transformers.DataCollatorWithPadding(tok)
  train_loader = DataLoader(ds['train'], batch_size=args.batch_size,
                            shuffle=True, collate_fn=collator)
  val_loader = DataLoader(ds['validation'], batch_size=args.batch_size * 2,
                          shuffle=False, collate_fn=collator)

  # Clean any stale weight files in the output dir (e.g. a single
  # model.safetensors left from a different model size) so the final save is
  # unambiguous — a leftover shard can otherwise be loaded against the wrong
  # config and silently fail.
  os.makedirs(args.output_dir, exist_ok=True)
  for f in (glob.glob(os.path.join(args.output_dir, 'model*.safetensors'))
            + glob.glob(os.path.join(args.output_dir, 'pytorch_model*.bin'))
            + glob.glob(os.path.join(args.output_dir, '*.index.json'))):
    os.remove(f)

  model = transformers.AutoModelForSequenceClassification.from_pretrained(
      args.model, num_labels=2, torch_dtype=torch.float32)
  model.config.pad_token_id = tok.pad_token_id
  model.gradient_checkpointing_enable()
  model.config.use_cache = False
  model = model.to(device)

  steps_per_epoch = (len(train_loader) + args.grad_accum - 1) // args.grad_accum
  total_steps = (args.max_steps if args.max_steps > 0
                 else int(args.epochs * steps_per_epoch))
  eval_steps = (args.eval_steps if args.eval_steps > 0
                else max(total_steps // 6, 1))
  optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
  sched = transformers.get_linear_schedule_with_warmup(
      optim, int(0.06 * total_steps), total_steps)
  print(f'[surrogate] model={args.model} total_steps={total_steps} '
        f'eval_steps={eval_steps} eff_batch={args.batch_size*args.grad_accum}',
        flush=True)

  best_f1 = -1.0
  step = 0
  micro = 0
  model.train()
  done = False

  def _maybe_eval_and_save(step):
    nonlocal best_f1
    m = _evaluate(model, val_loader, device)
    print(f'[surrogate] step {step}/{total_steps} val: '
          f"acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f} "
          f"prec={m['precision']:.4f} rec={m['recall']:.4f}", flush=True)
    if m['f1'] > best_f1:
      best_f1 = m['f1']
      model.save_pretrained(args.output_dir)
      tok.save_pretrained(args.output_dir)
      print(f'[surrogate]   ↑ new best F1={best_f1:.4f} — saved to '
            f'{args.output_dir}', flush=True)

  while not done:
    for batch in train_loader:
      labels = batch.pop('labels').to(device)
      batch = {k: v.to(device) for k, v in batch.items()}
      with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(**batch, labels=labels)
        loss = out.loss / args.grad_accum
      loss.backward()
      micro += 1
      if micro % args.grad_accum == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()
        optim.zero_grad()
        step += 1
        if step % 50 == 0:
          print(f'[surrogate] step {step}/{total_steps} '
                f'loss={loss.item()*args.grad_accum:.4f}', flush=True)
        if step % eval_steps == 0:
          _maybe_eval_and_save(step)
        if step >= total_steps:
          done = True
          break

  # Final eval; ensures a checkpoint exists even if F1 never improved.
  _maybe_eval_and_save(step)
  if best_f1 < 0:
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
  print(f'[train_toxic_surrogate] best val F1={best_f1:.4f}; saved to '
        f'{args.output_dir}', flush=True)


if __name__ == '__main__':
  main()
