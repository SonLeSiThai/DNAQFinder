"""Reference labels for empirical alignments with ModelFinder (IQ-TREE 2).

For every alignment the script runs

  iqtree2 -s <aln> -m MF -mset JC,K2P,F81,HKY,TN,GTR -mrate I+G4 \
          -keep-ident -T <threads> -pre <prefix> -mfreq ""

reads "Best-fit model according to BIC" from <prefix>.iqtree and maps the base
matrix to one of the 6 classes (JC, K2P, F81, HKY, TN, GTR). The IQ-TREE
outputs (.iqtree, .log, .treefile, ...) are kept; the .treefile of training
alignments is the tree input of simulation/alisim_mltree.py.

The labels of the paper were produced with this command using IQ-TREE 2.4.0
(training, validation, original internal test and TreeBASE) and IQ-TREE
2.3.6 (the 1,720 added internal-test alignments).

Note: for F81, HKY, TN and GTR the best model is expected to carry '+F'.
Some IQ-TREE versions also test equal-frequency variants of these matrices
and print them without '+F' (they are then JC, K2P, TNe or SYM). Such rows are
flagged in the column `freq_check` and should not be used as labels.

Example:
  python labeling/modelfinder_labels.py --fasta-dir empirical/train --out labels/train
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dnaqfinder import canon  # noqa: E402

MSET = 'JC,K2P,F81,HKY,TN,GTR'
MRATE = 'I+G4'
EXTS = ('*.fasta', '*.fa', '*.fas')
BEST_RE = re.compile(r'Best-fit model according to BIC:\s*(\S+)')


def run_one(iqtree, fasta, prefix, threads):
    cmd = [iqtree, '-s', fasta, '-m', 'MF', '-mset', MSET, '-mrate', MRATE,
           '-keep-ident', '-T', str(threads), '-pre', prefix, '-mfreq', '']
    if not os.path.exists(prefix + '.iqtree'):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(prefix + '.iqtree'):
        return None, 'iqtree_failed'
    with open(prefix + '.iqtree', encoding='utf-8', errors='replace') as fh:
        m = BEST_RE.search(fh.read())
    return (m.group(1), '') if m else (None, 'parse_failed')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fasta-dir', required=True, help='folder searched recursively for alignments')
    ap.add_argument('--out', required=True, help='output folder: labels.csv + iqtree/<sample>/')
    ap.add_argument('--iqtree', default='iqtree2')
    ap.add_argument('--threads', type=int, default=1, help='IQ-TREE threads per alignment (-T)')
    ap.add_argument('--jobs', type=int, default=1, help='alignments processed in parallel')
    args = ap.parse_args()

    files = sorted({f for e in EXTS for f in glob.glob(os.path.join(args.fasta_dir, '**', e), recursive=True)})
    if not files:
        sys.exit(f'no alignment found under {args.fasta_dir}')
    jobs = []
    for f in files:
        rel = os.path.relpath(f, args.fasta_dir)
        sample = os.path.splitext(rel)[0].replace(os.sep, '__')
        jobs.append((rel, os.path.join(args.out, 'iqtree', sample, sample)))

    rows = []
    with ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(run_one, args.iqtree, os.path.join(args.fasta_dir, rel), pre, args.threads): rel
                for rel, pre in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            rel = futs[fut]
            best, err = fut.result()
            label = canon(best) if best else ''
            if best and label in ('F81', 'HKY', 'TN', 'GTR') and '+F' not in best:
                freq = 'equal_freq_variant'
            else:
                freq = 'ok' if best else ''
            rows.append({'file': rel.replace(os.sep, '/'), 'modelfinder_best_model': best or '',
                         'label': label, 'freq_check': freq, 'status': err or 'ok'})
            if k % 100 == 0:
                print(f'  {k}/{len(jobs)}', file=sys.stderr)

    rows.sort(key=lambda r: r['file'])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'labels.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    n_ok = sum(r['status'] == 'ok' for r in rows)
    n_freq = sum(r['freq_check'] == 'equal_freq_variant' for r in rows)
    print(f'{n_ok}/{len(rows)} labelled; {n_freq} equal-frequency variants flagged -> {args.out}/labels.csv',
          file=sys.stderr)


if __name__ == '__main__':
    main()
