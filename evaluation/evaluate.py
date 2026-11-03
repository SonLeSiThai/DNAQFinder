"""Evaluate a checkpoint on labelled FASTA alignments.

Bundled test sets (data/*.zip, extracted on first use):
  seqgen_test   1,008 alignments simulated with the ModelRevelator pipeline;
                labels = generating model; score = accuracy
  treebase      1,144 empirical TreeBASE alignments; labels = ModelFinder
                (BIC); score = concordance with ModelFinder
For these two sets the result is compared with the values reported in the
paper. The released network is deterministic in eval mode, so the numbers
are expected to match to 4 decimals on any CPU.

Any other labelled set can be scored with --fasta-dir/--labels.

Examples:
  python evaluation/evaluate.py                      # both bundled sets
  python evaluation/evaluate.py --set treebase --save-pred out/treebase_pred.csv
  python evaluation/evaluate.py --fasta-dir sim/alisim_val/fasta \
      --labels sim/alisim_val/ground_truth.csv
"""
import argparse
import csv
import os
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import dnaqfinder as dq  # noqa: E402

DATA = os.path.join(ROOT, 'data')
BUNDLED = {
    # name: (folder, label csv, id col, label col, path col, reported values)
    'seqgen_test': ('seqgen_test_1008', 'ground_truth.csv', 'sample_id', 'true_label', '',
                    {'score': 0.8552, 'macro_f1': 0.8539, 'top2': 0.9613}),
    'treebase': ('treebase_1144', 'labels.csv', 'sample_id', 'label', 'file',
                 {'score': 0.5953, 'macro_f1': 0.5089, 'top2': 0.8208}),
}


def macro_f1(cm):
    fs = []
    for k in range(6):
        tp = cm[k, k]; fp = cm[:, k].sum() - tp; fn = cm[k].sum() - tp
        fs.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(fs))


def score(y, probs):
    order = np.argsort(-probs, axis=1); pred = order[:, 0]
    cm = np.zeros((6, 6), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    recall = {dq.CLASSES[k]: (cm[k, k] / cm[k].sum() if cm[k].sum() else None) for k in range(6)}
    return dict(n=len(y), score=float((pred == y).mean()), macro_f1=macro_f1(cm),
                top2=float(np.mean([y[i] in order[i, :2] for i in range(len(y))])),
                recall=recall, confusion=cm, pred=pred)


def load_set(fasta_dir, labels_csv, id_col, label_col, path_col):
    tab = pd.read_csv(labels_csv)
    ids, y, counts, missing = [], [], [], 0
    for row in tab.itertuples(index=False):
        lab = dq.cidx(getattr(row, label_col))
        sid = str(getattr(row, id_col))
        rel = str(getattr(row, path_col)) if path_col else sid + '.fasta'
        cc = dq.encode_fasta(os.path.join(fasta_dir, rel)) if lab >= 0 else None
        if cc is None:
            missing += 1; continue
        ids.append(sid); y.append(lab); counts.append(cc)
    return ids, np.asarray(y), counts, missing


def ensure_extracted(folder):
    path = os.path.join(DATA, folder)
    if not os.path.isdir(path):
        with zipfile.ZipFile(os.path.join(DATA, folder + '.zip')) as z:
            z.extractall(DATA)
    return path


def report(name, ids, y, probs, missing, reported, save_pred):
    r = score(y, probs)
    metric = 'concordance' if name == 'treebase' else 'accuracy'
    print(f'\n[{name}] n={r["n"]} (unreadable or unlabelled: {missing})')
    print(f'  {metric:<12} {r["score"]:.4f}')
    print(f'  {"macro-F1":<12} {r["macro_f1"]:.4f}')
    print(f'  {"top-2":<12} {r["top2"]:.4f}')
    print('  recall       ' + '  '.join(f'{c} {v:.3f}' if v is not None else f'{c} -'
                                       for c, v in r['recall'].items()))
    print('  confusion (rows = reference, columns = prediction; JC K2P F81 HKY TN GTR)')
    for k in range(6):
        print(f'    {dq.CLASSES[k]:<4}' + ''.join(f'{v:6d}' for v in r['confusion'][k]))
    ok = True
    if reported:
        for key, val in reported.items():
            match = abs(r[key] - val) < 1e-3
            ok &= match
            print(f'  paper {key:<9} {val:.4f}  {"match" if match else "DIFFERENT"}')
    if save_pred:
        os.makedirs(os.path.dirname(os.path.abspath(save_pred)), exist_ok=True)
        with open(save_pred, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['id', 'reference', 'prediction'] + [f'p_{c}' for c in dq.CLASSES])
            for i in range(len(ids)):
                w.writerow([ids[i], dq.CLASSES[y[i]], dq.CLASSES[r['pred'][i]]]
                           + [f'{v:.4f}' for v in probs[i]])
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', choices=['all', 'seqgen_test', 'treebase'], default='all')
    ap.add_argument('--ckpt', default=dq.DEFAULT_CKPT)
    ap.add_argument('--pool', choices=['mean', 'attn'], default='mean')
    ap.add_argument('--fasta-dir', help='custom set: folder with the alignments')
    ap.add_argument('--labels', help='custom set: CSV with ids and labels')
    ap.add_argument('--id-col', default='sample_id')
    ap.add_argument('--label-col', default='true_label')
    ap.add_argument('--path-col', default='', help='CSV column with the FASTA path relative to --fasta-dir')
    ap.add_argument('--save-pred', default='', help='write per-alignment predictions to this CSV')
    args = ap.parse_args()

    model = dq.load_model(args.ckpt, args.pool)
    all_ok = True
    if args.fasta_dir:
        ids, y, counts, missing = load_set(args.fasta_dir, args.labels, args.id_col, args.label_col, args.path_col)
        probs = dq.predict_counts(model, counts)[1]
        report('custom', ids, y, probs, missing, None, args.save_pred)
        return
    names = list(BUNDLED) if args.set == 'all' else [args.set]
    released = os.path.abspath(args.ckpt) == os.path.abspath(dq.DEFAULT_CKPT)
    for name in names:
        folder, lab_csv, id_col, label_col, path_col, reported = BUNDLED[name]
        base = ensure_extracted(folder)
        ids, y, counts, missing = load_set(os.path.join(base, 'fasta'), os.path.join(base, lab_csv),
                                           id_col, label_col, path_col)
        probs = dq.predict_counts(model, counts)[1]
        save = args.save_pred if len(names) == 1 else (
            args.save_pred and os.path.splitext(args.save_pred)[0] + f'_{name}.csv')
        all_ok &= report(name, ids, y, probs, missing, reported if released else None, save)
    if released:
        print('\nall values match the paper' if all_ok else '\nsome values differ from the paper')
        sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
