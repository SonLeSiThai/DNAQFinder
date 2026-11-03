"""Simulate training alignments with AliSim on random Yule-Harding trees.

Every alignment is simulated under one of the 6 substitution models with
+I+G4 rate heterogeneity. The number of sequences, the alignment length,
p_inv and alpha are drawn jointly from regime_pool.npz (100,000 tuples; see
README). The remaining parameters are drawn per alignment:
  kappa (K2P, HKY; both TN rates)  log-uniform [1, 15]
  GTR exchangeabilities            log-uniform [0.3, 8], 6 rates
  base frequencies (F81..GTR)      Dirichlet(a, a, a, a), a ~ U(0.5, 6),
                                   clipped at 0.05 and renormalised
  tree                             RANDOM{yh/n} (Yule-Harding)
  branch scale                     log-uniform [0.02, 0.6]
Labels cycle JC, K2P, F81, HKY, TN, GTR, so the classes are balanced.

Example:
  python simulation/alisim_yule.py --n 18000 --seed 7 --out sim/alisim_train
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POOL = os.path.join(HERE, 'regime_pool.npz')
MODELS = ['JC', 'K2P', 'F81', 'HKY', 'TN', 'GTR']


def load_regime(pool_path):
    """(n_sequences, length, p_inv, alpha) sampling pool."""
    d = np.load(pool_path)
    return d['taxa'], d['length'], d['pinv'], d['alpha']


def model_spec(model, pinv, alpha, rng):
    """AliSim model string, e.g. 'HKY{3.214}+F{0.31/0.22/0.18/0.29}+I{0.120}+G4{0.850}'."""
    rate = f"+I{{{pinv:.3f}}}+G4{{{alpha:.3f}}}"
    kap = lambda: float(np.exp(rng.uniform(np.log(1.0), np.log(15.0))))
    if model in ('JC', 'K2P'):
        freqs = ''
    else:
        pi = rng.dirichlet([rng.uniform(0.5, 6)] * 4); pi = np.clip(pi, 0.05, None); pi /= pi.sum()
        freqs = '+F{%.3f/%.3f/%.3f/%.3f}' % tuple(pi)
    if model == 'JC':
        base = 'JC'
    elif model == 'K2P':
        base = 'K80{%.3f}' % kap()
    elif model == 'F81':
        base = 'F81'
    elif model == 'HKY':
        base = 'HKY{%.3f}' % kap()
    elif model == 'TN':
        base = 'TN{%.3f/%.3f}' % (kap(), kap())
    else:
        r = np.exp(rng.uniform(np.log(0.3), np.log(8.0), 6))
        base = 'GTR{%.3f/%.3f/%.3f/%.3f/%.3f/%.3f}' % tuple(r)
    return base + freqs + rate


def parse_fasta(path):
    recs, name, seq = [], None, []
    with open(path) as fh:
        for ln in fh:
            ln = ln.rstrip('\n')
            if ln.startswith('>'):
                if name is not None:
                    recs.append((name, ''.join(seq)))
                name = ln[1:].strip().split()[0]; seq = []
            elif ln.strip():
                seq.append(ln.strip())
    if name is not None:
        recs.append((name, ''.join(seq)))
    return recs


def gen_one(iqtree, i, model, taxa, length, pinv, alpha, work_dir, out_fasta_dir, seed):
    rng = np.random.default_rng(seed * 1_000_003 + i)
    bscale = float(np.exp(rng.uniform(np.log(0.02), np.log(0.6))))
    spec = model_spec(model, pinv, alpha, rng)
    pre = os.path.join(work_dir, f'g{i:06d}')
    try:
        subprocess.run([iqtree, '--alisim', pre, '-t', f'RANDOM{{yh/{taxa}}}', '-m', spec,
                        '--length', str(length), '--branch-scale', f'{bscale:.4f}',
                        '-af', 'fasta', '-T', '1'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        fa = pre + '.fa'
        if not os.path.exists(fa):
            return None
        recs = parse_fasta(fa)
        if len(recs) != taxa:
            return None
        sid = f'alisim{i:05d}_{model}_{taxa}t_{length}bp_Inv{pinv:.2f}_Gam{alpha:.2f}'
        with open(os.path.join(out_fasta_dir, sid + '.fasta'), 'w') as f:
            for nm, sq in recs:
                f.write(f'>{nm}\n{sq}\n')
        return {'sample_id': sid, 'true_label': model, 'num_taxa': int(taxa),
                'msa_length': int(length), 'pinv': float(pinv), 'alpha': float(alpha),
                'branch_scale': bscale, 'spec': spec}
    except Exception:
        return None
    finally:
        for f in glob.glob(pre + '.*'):
            try:
                os.remove(f)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, required=True, help='number of alignments')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', required=True, help='output folder (fasta/ + ground_truth.csv)')
    ap.add_argument('--iqtree', default='iqtree2', help='IQ-TREE 2 executable (AliSim)')
    ap.add_argument('--pool', default=DEFAULT_POOL, help='regime pool .npz')
    ap.add_argument('--workers', type=int, default=0, help='parallel AliSim processes (0 = CPUs - 1)')
    args = ap.parse_args()

    taxa_pool, len_pool, pinv_pool, alpha_pool = load_regime(args.pool)
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(taxa_pool), args.n)

    out_fasta_dir = os.path.join(args.out, 'fasta')
    os.makedirs(out_fasta_dir, exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix='alisim_work_')
    nw = args.workers or max(2, (os.cpu_count() or 4) - 1)

    jobs = []
    for i in range(args.n):
        j = idx[i]
        jobs.append((args.iqtree, i, MODELS[i % 6], int(taxa_pool[j]), int(len_pool[j]),
                     float(pinv_pool[j]), float(alpha_pool[j]), work_dir, out_fasta_dir, args.seed))

    print(f'AliSim (Yule-Harding): n={args.n} seed={args.seed} workers={nw} -> {out_fasta_dir}', file=sys.stderr)
    rows, t0, c, fail = [], time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=nw) as ex:
        futs = [ex.submit(gen_one, *jb) for jb in jobs]
        for fut in as_completed(futs):
            r = fut.result(); c += 1
            if r is None:
                fail += 1
            else:
                rows.append(r)
            if c % 200 == 0:
                rate = (time.time() - t0) / c
                print(f'  {c}/{args.n}  ok={len(rows)} fail={fail}  {rate:.3f} s/aln', file=sys.stderr)
    try:
        os.rmdir(work_dir)
    except OSError:
        pass

    gt = pd.DataFrame(rows)
    gt.to_csv(os.path.join(args.out, 'ground_truth.csv'), index=False)
    print(f'done: ok={len(gt)} fail={fail} time={(time.time() - t0) / 60:.1f} min', file=sys.stderr)
    if len(gt):
        print(gt.true_label.value_counts().sort_index().to_string(), file=sys.stderr)


if __name__ == '__main__':
    main()
