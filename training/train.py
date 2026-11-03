"""Train the DNAQFinder Set Transformer on encoded alignments (.npz).

Defaults reproduce the training of the released model (model/best_mean.pt):
  training set   7 simulated files (222,000 alignments) + emp_train.npz
                 (15,343 empirical alignments) repeated 4 times by index,
                 i.e. 283,372 examples per epoch, shuffled together
  validation     emp_val.npz (1,856 empirical alignments); the checkpoint
                 with the highest validation accuracy is kept
  network        pool='mean', 553,896 parameters, initialised with seed 0
  optimiser      AdamW, lr 3e-4, weight decay 1e-4, batch 128 (drop_last)
  schedule       linear warm-up over 3 epochs, then cosine decay; 30 epochs
                 (the learning rate is updated once per epoch)
  loss           cross-entropy with label smoothing 0.05; gradient norm clip 1.0
  columns        800 columns drawn at random without replacement per example
                 and epoch (training); 1,500 evenly spaced columns (evaluation)

Example (GPU recommended):
  python training/train.py --data-dir encoded --out runs/m_p3
Quick check on CPU (8 examples per class, 1 epoch; a small batch is needed
because incomplete batches are dropped):
  python training/train.py --data-dir encoded --out runs/smoke --smoke 8 --epochs 1 --bs 8 --workers 0
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dnaqfinder import CLASSES, SetTransformer  # noqa: E402

SIM_TRAIN = ['mr_sim_train.npz', 'mr_sim_train_b.npz', 'mr_sim_train_c.npz', 'mr_sim_train_d.npz',
             'alisim_train.npz', 'alisim_train2.npz', 'alisim_train_rt.npz']
EMP_TRAIN = ['emp_train.npz']
VAL = ['emp_val.npz']


class DS(Dataset):
    """specs: list of (npz path, repeat). Each file is loaded once; a repeat > 1
    oversamples that file by repeating its indices."""
    def __init__(self, specs, max_cols, train):
        cc, off, lab = [], [0], []
        self.index = []
        for path, rep in specs:
            d = np.load(path, allow_pickle=True)
            base = off[-1]; start = len(lab); n_here = len(d['labels'])
            cc.append(d['counts'])
            off.extend((d['offsets'][1:] + base).tolist()); lab.extend(d['labels'].tolist())
            self.index.extend(list(range(start, start + n_here)) * rep)
        self.counts = np.concatenate(cc); self.offsets = np.asarray(off, np.int64)
        self.labels = np.asarray(lab, np.int8); self.max_cols = max_cols; self.train = train
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i = self.index[k]
        cc = self.counts[self.offsets[i]:self.offsets[i + 1]]; L = cc.shape[0]
        if self.max_cols and L > self.max_cols:
            sel = self.rng.choice(L, self.max_cols, replace=False) if self.train \
                else np.linspace(0, L - 1, self.max_cols).astype(int)
            cc = cc[np.sort(sel)]
        return torch.from_numpy(cc.astype(np.float32)), int(self.labels[i])


def collate(b):
    seqs, labs = zip(*b); Lmax = max(s.shape[0] for s in seqs); B = len(seqs)
    x = torch.zeros(B, Lmax, 5); mask = torch.zeros(B, Lmax, dtype=torch.bool)
    for j, s in enumerate(seqs):
        x[j, :s.shape[0]] = s; mask[j, :s.shape[0]] = True
    return x, mask, torch.tensor(labs)


@torch.no_grad()
def predict(model, loader, dev):
    model.eval(); ys, ps = [], []
    for x, mask, y in loader:
        ps.append(model(x.to(dev), mask.to(dev)).argmax(1).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def metrics(y, p):
    cm = np.zeros((6, 6), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    rec = {CLASSES[c]: (round(float(cm[c, c] / cm[c].sum()), 3) if cm[c].sum() else None) for c in range(6)}
    return float((y == p).mean()), rec, cm.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True, help='folder with the .npz files')
    ap.add_argument('--out', required=True, help='output folder for best_<pool>.pt and metrics.json')
    ap.add_argument('--sim-train', nargs='+', default=SIM_TRAIN)
    ap.add_argument('--emp-train', nargs='*', default=EMP_TRAIN)
    ap.add_argument('--emp-oversample', type=int, default=4)
    ap.add_argument('--val', nargs='+', default=VAL, help='checkpoint selection set(s)')
    ap.add_argument('--test', nargs='*', default=[], help='optional .npz test sets scored after training')
    ap.add_argument('--pool', choices=['mean', 'attn'], default='mean')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--train-max-cols', type=int, default=800)
    ap.add_argument('--eval-max-cols', type=int, default=1500)
    ap.add_argument('--workers', type=int, default=2, help='DataLoader workers (use 0 on Windows)')
    ap.add_argument('--smoke', type=int, default=0, help='keep only this many examples per class')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    p = lambda f: os.path.join(args.data_dir, f)
    train_specs = [(p(f), 1) for f in args.sim_train] + [(p(f), args.emp_oversample) for f in args.emp_train]
    tr = DS(train_specs, args.train_max_cols, True)
    val = DS([(p(f), 1) for f in args.val], args.eval_max_cols, False)
    tests = {os.path.splitext(f)[0]: DS([(p(f), 1)], args.eval_max_cols, False) for f in args.test}
    if args.smoke:
        def sub(d, n):
            idx = np.concatenate([np.where(d.labels == c)[0][:n] for c in range(6)])
            return Subset(d, idx.tolist())
        tr = sub(tr, args.smoke); val = sub(val, max(8, args.smoke // 4))
        tests = {k: sub(d, max(8, args.smoke // 4)) for k, d in tests.items()}
    print(f'device={dev} train={len(tr)} val={len(val)} tests={ {k: len(d) for k, d in tests.items()} }')

    torch.manual_seed(0)
    model = SetTransformer(args.pool).to(dev)
    n_params = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_at(ep):
        if ep < args.warmup:
            return (ep + 1) / args.warmup
        prog = (ep - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    tl = DataLoader(tr, args.bs, shuffle=True, collate_fn=collate, num_workers=args.workers,
                    drop_last=True, pin_memory=(dev == 'cuda'))
    vl = DataLoader(val, 256, collate_fn=collate, num_workers=args.workers)

    ckpt = os.path.join(args.out, f'best_{args.pool}.pt')
    best, history = -1.0, []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, mask, y in tl:
            opt.zero_grad()
            loss = lossf(model(x.to(dev), mask.to(dev)), y.to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step(); tot += loss.item()
        sch.step()
        yv, pv = predict(model, vl, dev); va = float((yv == pv).mean())
        if va > best:
            best = va; torch.save(model.state_dict(), ckpt)
        history.append({'epoch': ep + 1, 'loss': tot / max(1, len(tl)), 'val_acc': va})
        print(f'epoch {ep + 1:02d} loss={tot / max(1, len(tl)):.4f} val_acc={va:.4f} best={best:.4f} '
              f'({time.time() - t0:.0f} s)', flush=True)

    model.load_state_dict(torch.load(ckpt, map_location=dev))
    res = {'pool': args.pool, 'params': n_params, 'val_best': best, 'history': history,
           'config': {k: v for k, v in vars(args).items()}, 'tests': {}}
    for name, d in tests.items():
        y, pr = predict(model, DataLoader(d, 256, collate_fn=collate, num_workers=args.workers), dev)
        acc, rec, cm = metrics(y, pr)
        res['tests'][name] = {'acc': round(acc, 4), 'recall': rec, 'confusion': cm}
        print(f'{name}: accuracy={acc:.4f} recall={rec}')
    with open(os.path.join(args.out, 'metrics.json'), 'w') as fh:
        json.dump(res, fh, indent=2)
    print(f'best validation accuracy {best:.4f}; checkpoint {ckpt}')


if __name__ == '__main__':
    main()
