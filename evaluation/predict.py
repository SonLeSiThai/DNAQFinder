"""Predict the substitution model of new FASTA alignments.

Input: one FASTA file or a folder (searched recursively for *.fasta, *.fa,
*.fas, *.aln). Output: the predicted class and the probabilities of the 6
classes (JC, K2P, F81, HKY, TN, GTR) for each alignment.

Examples:
  python evaluation/predict.py my_alignment.fasta
  python evaluation/predict.py my_alignments/ -o predictions.csv
"""
import argparse
import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import dnaqfinder as dq  # noqa: E402


def gather(path):
    if os.path.isdir(path):
        files = []
        for ext in ('*.fasta', '*.fa', '*.fas', '*.aln'):
            files += glob.glob(os.path.join(path, '**', ext), recursive=True)
        return sorted(files)
    return [path]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='a FASTA file or a folder of alignments')
    ap.add_argument('-o', '--out', help='write the results to this CSV')
    ap.add_argument('--ckpt', default=dq.DEFAULT_CKPT)
    args = ap.parse_args()

    files = gather(args.input)
    if not files:
        sys.exit(f'no alignment found under {args.input}')
    model = dq.load_model(args.ckpt)
    kept, counts, bad = [], [], []
    for f in files:
        cc = dq.encode_fasta(f)
        if cc is None:
            bad.append(f)
        else:
            kept.append(f); counts.append(cc)
    if not kept:
        sys.exit('no readable alignment')
    pred, probs, _, t_per = dq.predict_counts(model, counts)

    print(f'{"alignment":<48}{"model":>8}   p(JC K2P F81 HKY TN GTR)')
    rows = []
    for i, f in enumerate(kept):
        print(f'{os.path.basename(f):<48}{dq.CLASSES[pred[i]]:>8}   ' + ' '.join(f'{v:.3f}' for v in probs[i]))
        rows.append([f, dq.CLASSES[pred[i]]] + [f'{v:.4f}' for v in probs[i]])
    if bad:
        print(f'skipped {len(bad)} unreadable file(s)', file=sys.stderr)
    print(f'{len(kept)} alignment(s); mean network time {t_per * 1000:.1f} ms per alignment')
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['fasta', 'prediction'] + [f'p_{c}' for c in dq.CLASSES])
            w.writerows(rows)


if __name__ == '__main__':
    main()
