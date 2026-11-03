"""Simulate alignments with Seq-Gen following the ModelRevelator pipeline.

Re-implementation of the simulation part of ModelRevelator's
src/simulate_summarystats.py (Burgstaller-Muehlbacher et al., 2023), without
the feature extraction and with the alignment written to FASTA:
  * exchangeabilities and base frequencies are sampled from the empirical
    distributions of the Lanfear parameters (values < 10, astroML
    EmpiricalDistribution); rGT is fixed at 1;
  * trees with 8, 16, 64 or 128 sequences are taken from trees.csv.gz and
    their branch lengths are resampled from the Lanfear branch-length
    distributions (redo_tree), rejecting lengths <= 1/L;
  * sequences are evolved with seq-gen using ModelRevelator's command templates;
  * the work list is balanced over 6 models x {no +G, +G} x 4 tree sizes;
    +G cells cycle through the alpha grid below; no +I.
The class label is the base model; +G and alpha are recorded separately.

Requires the files trees.csv.gz, parameters_lanfear.tsv.gz and
branchlen_lanfear.tsv.gz from the simulation_parameters/ folder of
https://github.com/Cibiv/ModelRevelator (MIT licence).

Example:
  python simulation/seqgen_modelrevelator.py --n 1008 --seed 20260625 \
      --params-dir ModelRevelator/simulation_parameters --out sim/seqgen_test_1008
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from astroML.density_estimation import EmpiricalDistribution

# ModelRevelator's seq-gen command templates (simulate_summarystats.py).
# The 'seq-gen' token is replaced by the executable; '-op' (PHYLIP output) is added.
MODELS = {
    'JC':     'seq-gen -mGTR -r1,1,1,1,1,1 -f0.25,0.25,0.25,0.25 -l{seq_len}',
    'K2P':    'seq-gen -mGTR -r1,{rAG},1,1,{rAG},1 -f0.25,0.25,0.25,0.25 -l{seq_len}',
    'F81':    'seq-gen -mGTR -r1,1,1,1,1,1 -f{piA},{piC},{piG},{piT} -l{seq_len}',
    'HKY':    'seq-gen -mGTR -r1,{rAG},1,1,{rAG},1 -f{piA},{piC},{piG},{piT} -l{seq_len}',
    'TN93':   'seq-gen -mGTR -r1,{rAG},1,1,{rCT},1 -f{piA},{piC},{piG},{piT} -l{seq_len}',
    'GTR':    'seq-gen -mGTR -r{rAC},{rAG},{rAT},{rCG},{rCT},1 -f{piA},{piC},{piG},{piT} -l{seq_len}',
    'JC+G':   'seq-gen -mGTR -r1,1,1,1,1,1 -f0.25,0.25,0.25,0.25 -l{seq_len} -a {alpha}',
    'K2P+G':  'seq-gen -mGTR -r1,{rAG},1,1,{rAG},1 -f0.25,0.25,0.25,0.25 -l{seq_len} -a {alpha}',
    'F81+G':  'seq-gen -mGTR -r1,1,1,1,1,1 -f{piA},{piC},{piG},{piT} -l{seq_len} -a {alpha}',
    'HKY+G':  'seq-gen -mGTR -r1,{rAG},1,1,{rAG},1 -f{piA},{piC},{piG},{piT} -l{seq_len} -a {alpha}',
    'TN93+G': 'seq-gen -mGTR -r1,{rAG},1,1,{rCT},1 -f{piA},{piC},{piG},{piT} -l{seq_len} -a {alpha}',
    'GTR+G':  'seq-gen -mGTR -r{rAC},{rAG},{rAT},{rCG},{rCT},1 -f{piA},{piC},{piG},{piT} -l{seq_len} -a {alpha}',
}
ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
TAXA = [8, 16, 64, 128]
BASE_MODELS = ['JC', 'K2P', 'F81', 'HKY', 'TN93', 'GTR']

SEQ_GEN = 'seq-gen'
lanfear_eds = {}
trees_by_taxa = {}


def load_params(params_dir):
    """Build the empirical distributions and the tree pool (no random draws here)."""
    trees_df = pd.read_csv(os.path.join(params_dir, 'trees.csv.gz'), compression='gzip')
    params = pd.read_csv(os.path.join(params_dir, 'parameters_lanfear.tsv.gz'), compression='gzip', sep='\t')
    blen = pd.read_csv(os.path.join(params_dir, 'branchlen_lanfear.tsv.gz'), compression='gzip', sep='\t')
    for col, name in [('AC', 'rAC'), ('AG', 'rAG'), ('AT', 'rAT'), ('CG', 'rCG'), ('CT', 'rCT'),
                      ('GT', 'rGT'), ('fA', 'piA'), ('fC', 'piC'), ('fG', 'piG'), ('fT', 'piT')]:
        lanfear_eds[name] = EmpiricalDistribution(params[col].loc[params[col] < 10])
    lanfear_eds['external_branchlen'] = EmpiricalDistribution(blen['branchlen'].loc[blen.extint == 'e'])
    lanfear_eds['internal_branchlen'] = EmpiricalDistribution(blen['branchlen'].loc[blen.extint == 'i'])
    for t in TAXA:
        trees_by_taxa[t] = trees_df[trees_df.nTaxa == t].reset_index(drop=True)


def redo_tree(tree, msa_length):
    """Resample internal and external branch lengths from the Lanfear distributions."""
    p = re.compile('[^t0-9]:[0-9.e-]*')
    fragments, last_end = [], 0
    for x in p.finditer(tree):
        r, l = x.span()
        r += 2
        bl = 0
        while bl == 0:
            t = lanfear_eds['internal_branchlen'].rvs(1)[0]
            if t > 1 / msa_length:
                bl = t
        fragments.append(tree[last_end:r] + f'{bl:.11f}')
        last_end = l
    fragments.append(tree[last_end:])
    tree = ''.join(fragments)

    p = re.compile('t[0-9]*:[0-9.e-]*')
    fragments, last_end = [], 0
    for x in p.finditer(tree):
        r, l = x.span()
        taxon_label = tree[r:l].split(':')[0]
        bl = 0
        while bl == 0:
            t = lanfear_eds['external_branchlen'].rvs(1)[0]
            if t > 1 / msa_length:
                bl = t
        fragments.append(tree[last_end:r] + taxon_label + ':' + f'{bl:.11f}')
        last_end = l
    fragments.append(tree[last_end:])
    return ''.join(fragments)


def run_seqgen(full_model, tree, seq_len, alpha):
    """Fill the command template with fresh parameter draws and run seq-gen.

    The tree is passed on stdin from a file (piping a string fails on Windows)."""
    rates_freq = {k: v.rvs(1)[0] for k, v in lanfear_eds.items()}
    args = MODELS[full_model].format(**rates_freq, seq_len=seq_len, alpha=alpha).split()
    argv = [SEQ_GEN] + args[1:] + ['-op']
    with tempfile.NamedTemporaryFile('w', suffix='.nwk', delete=False) as tf:
        tf.write(tree + '\n')
        tree_path = tf.name
    try:
        with open(tree_path, 'rb') as fh:
            proc = subprocess.run(argv, stdin=fh, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    finally:
        os.unlink(tree_path)
    return proc.stdout.decode()


def parse_phylip(text):
    """Relaxed PHYLIP from seq-gen -> list of (name, sequence)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    recs = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) >= 2:
            recs.append((parts[0], parts[-1]))
    return recs


def simulate_one(full_model, taxa, seq_len, alpha, rng, max_tries=20):
    pool = trees_by_taxa[taxa]
    for _ in range(max_tries):
        row = pool.iloc[rng.integers(0, len(pool))]
        tree = redo_tree(row['treeNewick'], seq_len)
        recs = parse_phylip(run_seqgen(full_model, tree, seq_len, alpha))
        if len(recs) == taxa:
            return recs, int(row['seed']), float(row['treeLength'])
    raise RuntimeError(f'seq-gen failed for {full_model}, {taxa} sequences, after {max_tries} tries')


def build_worklist(n_target):
    """Balanced over base model x {no +G, +G} x tree size; +G cells cycle the alpha grid."""
    cells = [(bm + suf, tx) for bm in BASE_MODELS for suf in ('', '+G') for tx in TAXA]
    per_cell = max(1, round(n_target / len(cells)))
    work, ai = [], 0
    for full_model, taxa in cells:
        for _ in range(per_cell):
            if full_model.endswith('+G'):
                alpha = ALPHAS[ai % len(ALPHAS)]
                ai += 1
            else:
                alpha = 0
            work.append((full_model, taxa, alpha))
    return work


def main():
    global SEQ_GEN
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, required=True, help='target number of alignments (rounded to a multiple of 48)')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--seq-len', type=int, default=1000)
    ap.add_argument('--params-dir', required=True, help='ModelRevelator simulation_parameters/ folder')
    ap.add_argument('--seqgen', default='seq-gen', help='Seq-Gen executable')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    SEQ_GEN = args.seqgen

    load_params(args.params_dir)
    fasta_dir = os.path.join(args.out, 'fasta')
    os.makedirs(fasta_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)          # astroML EmpiricalDistribution.rvs uses the global NumPy RNG

    work = build_worklist(args.n)
    print(f'Seq-Gen: {len(work)} alignments, {args.seq_len} sites -> {fasta_dir}', file=sys.stderr)
    rows = []
    for i, (full_model, taxa, alpha) in enumerate(work):
        recs, tree_seed, treelen = simulate_one(full_model, taxa, args.seq_len, alpha, rng)
        a_str = f'_a{alpha}' if alpha else ''
        sample_id = f"sim{i:04d}_{full_model.replace('+', '')}_{taxa}t_{args.seq_len}bp{a_str}"
        with open(os.path.join(fasta_dir, sample_id + '.fasta'), 'w') as f:
            for name, seq in recs:
                f.write(f'>{name}\n{seq}\n')
        rows.append({'sample_id': sample_id, 'true_label': full_model.replace('+G', ''),
                     'full_model': full_model, 'has_gamma': full_model.endswith('+G'),
                     'alpha': alpha, 'num_taxa': taxa, 'seq_len': args.seq_len,
                     'tree_seed': tree_seed, 'tree_length': treelen})
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{len(work)}', file=sys.stderr)

    gt = pd.DataFrame(rows)
    gt.to_csv(os.path.join(args.out, 'ground_truth.csv'), index=False)
    print(f'done: {len(gt)} alignments, +G fraction {gt.has_gamma.mean():.3f}', file=sys.stderr)
    print(gt.true_label.value_counts().sort_index().to_string(), file=sys.stderr)


if __name__ == '__main__':
    main()
