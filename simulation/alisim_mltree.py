"""Simulate training alignments with AliSim on maximum-likelihood trees.

Same substitution and rate parameters as alisim_yule.py, but each alignment
evolves along an ML tree (.treefile) inferred by IQ-TREE from an empirical
training alignment (produced by labeling/modelfinder_labels.py). The tree
gives the topology, the number of sequences and the branch lengths; branch
lengths are multiplied by a factor drawn log-uniformly from [0.7, 1.4].
Length, p_inv and alpha are drawn from regime_pool.npz.

Use trees of the training split for the training set and trees of the
validation split for the validation set, so that no test alignment
contributes a tree.

Example:
  python simulation/alisim_mltree.py --n 60000 --seed 11 \
      --tree-dir labels/train/iqtree --out sim/alisim_rt_train
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from alisim_yule import DEFAULT_POOL, MODELS, load_regime, model_spec, parse_fasta  # noqa: E402


def tree_pool(tree_dir):
    """All .treefile files under tree_dir (recursive), sorted by path."""
    return sorted(glob.glob(os.path.join(tree_dir, '**', '*.treefile'), recursive=True))


def gen_one(iqtree, i, model, treefile, length, pinv, alpha, work_dir, out_fasta_dir, seed):
    rng = np.random.default_rng(seed * 1_000_003 + i)
    bscale = float(np.exp(rng.uniform(np.log(0.7), np.log(1.4))))
    spec = model_spec(model, pinv, alpha, rng)
    pre = os.path.join(work_dir, f'g{i:06d}')
    try:
        tloc = pre + '.treefile'          # copy: IQ-TREE on Windows fails on non-ASCII tree paths
        shutil.copy(treefile, tloc)
        subprocess.run([iqtree, '--alisim', pre, '-t', tloc, '-m', spec,
                        '--length', str(length), '--branch-scale', f'{bscale:.4f}',
                        '-af', 'fasta', '-T', '1'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        fa = pre + '.fa'
        if not os.path.exists(fa):
            return None
        recs = parse_fasta(fa)
        if len(recs) < 4:
            return None
        taxa = len(recs)
        sid = f'alirt{i:05d}_{model}_{taxa}t_{length}bp_Inv{pinv:.2f}_Gam{alpha:.2f}'
        with open(os.path.join(out_fasta_dir, sid + '.fasta'), 'w') as f:
            for nm, sq in recs:
                f.write(f'>{nm}\n{sq}\n')
        return {'sample_id': sid, 'true_label': model, 'num_taxa': int(taxa),
                'msa_length': int(length), 'pinv': float(pinv), 'alpha': float(alpha),
                'branch_scale': bscale, 'tree': os.path.basename(treefile)}
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
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--tree-dir', required=True, help='folder searched recursively for *.treefile')
    ap.add_argument('--out', required=True)
    ap.add_argument('--iqtree', default='iqtree2')
    ap.add_argument('--pool', default=DEFAULT_POOL)
    ap.add_argument('--workers', type=int, default=0)
    args = ap.parse_args()

    _, len_pool, pinv_pool, alpha_pool = load_regime(args.pool)
    trees = tree_pool(args.tree_dir)
    if not trees:
        sys.exit(f'no .treefile found under {args.tree_dir}')
    rng = np.random.default_rng(args.seed)
    tree_idx = rng.integers(0, len(trees), args.n)
    reg_idx = rng.integers(0, len(len_pool), args.n)

    out_fasta_dir = os.path.join(args.out, 'fasta')
    os.makedirs(out_fasta_dir, exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix='alirt_work_')
    nw = args.workers or max(2, (os.cpu_count() or 4) - 1)

    jobs = []
    for i in range(args.n):
        j = reg_idx[i]
        jobs.append((args.iqtree, i, MODELS[i % 6], trees[tree_idx[i]], int(len_pool[j]),
                     float(pinv_pool[j]), float(alpha_pool[j]), work_dir, out_fasta_dir, args.seed))

    print(f'AliSim (ML trees): {len(trees)} trees, n={args.n} seed={args.seed} workers={nw}', file=sys.stderr)
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
