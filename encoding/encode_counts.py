"""Encode a set of FASTA alignments as per-column base counts (.npz).

Each alignment becomes an int16 array [L, 5] of counts of A, C, G, T and
other characters (gap, N, IUPAC codes) per column. A dataset is stored packed:
  counts  [sum L, 5] int16   all alignments concatenated along the columns
  offsets [N + 1]    int64   alignment i = counts[offsets[i]:offsets[i+1]]
  labels  [N]        int8    0..5 = JC, K2P, F81, HKY, TN, GTR
  lengths [N], ntaxa [N], ids [N]
The 17 per-column features are computed from these counts inside the network
(dnaqfinder.FrontEnd), so this file is the only preprocessing step.

The label table is a CSV with an id column and a label column. The FASTA path
is either <fasta-dir>/<id>.fasta (simulated sets) or given by --path-col,
relative to --fasta-dir (empirical sets labelled by modelfinder_labels.py).
Rows whose label is not one of the 6 classes are skipped.

Examples:
  python encoding/encode_counts.py --fasta-dir sim/alisim_train/fasta \
      --labels sim/alisim_train/ground_truth.csv --out encoded/alisim_train.npz
  python encoding/encode_counts.py --fasta-dir empirical/train \
      --labels labels/train/labels.csv --id-col file --label-col label \
      --path-col file --out encoded/emp_train.npz
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dnaqfinder import cidx, col_counts, read_fasta_codes  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fasta-dir', required=True)
    ap.add_argument('--labels', required=True, help='CSV with ids and labels')
    ap.add_argument('--out', required=True, help='output .npz')
    ap.add_argument('--id-col', default='sample_id')
    ap.add_argument('--label-col', default='true_label')
    ap.add_argument('--path-col', default='', help='CSV column holding the FASTA path relative to --fasta-dir')
    args = ap.parse_args()

    tab = pd.read_csv(args.labels)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    counts, offsets, labels, lengths, ntaxa, ids = [], [0], [], [], [], []
    skipped, t0 = 0, time.time()
    for n, row in enumerate(tab.itertuples(index=False)):
        sid = str(getattr(row, args.id_col))
        lab = cidx(getattr(row, args.label_col))
        if lab < 0:
            skipped += 1; continue
        if args.path_col:
            path = os.path.join(args.fasta_dir, str(getattr(row, args.path_col)))
        else:
            path = os.path.join(args.fasta_dir, sid + '.fasta')
        mat = read_fasta_codes(path) if os.path.exists(path) else None
        if mat is None:
            skipped += 1; continue
        cc = col_counts(mat)
        counts.append(cc)
        offsets.append(offsets[-1] + cc.shape[0])
        labels.append(lab); lengths.append(cc.shape[0]); ntaxa.append(mat.shape[0]); ids.append(sid)
        if (n + 1) % 2000 == 0:
            print(f'  {n + 1}/{len(tab)}  encoded={len(labels)} skipped={skipped}  {time.time() - t0:.0f} s',
                  file=sys.stderr)

    np.savez(args.out,
             counts=np.concatenate(counts, axis=0).astype(np.int16),
             offsets=np.asarray(offsets, np.int64),
             labels=np.asarray(labels, np.int8),
             lengths=np.asarray(lengths, np.int32),
             ntaxa=np.asarray(ntaxa, np.int32),
             ids=np.asarray(ids))
    print(f'wrote {args.out}: N={len(labels)} skipped={skipped} '
          f'class counts={np.bincount(labels, minlength=6).tolist()}', file=sys.stderr)


if __name__ == '__main__':
    main()
