# DNAQFinder

DNAQFinder predicts which of six nucleotide substitution models (JC, K2P, F81, HKY, TN93, GTR) fits a DNA multiple sequence alignment. It counts the bases in every alignment column and computes 17 statistics per column from these counts. A Set Transformer classifies the resulting set of columns. The method does not reconstruct a tree and does not compute likelihoods.

The network was trained on 222,000 simulated alignments and 15,343 empirical alignments. The empirical alignments are labelled with the model selected by ModelFinder under BIC.

This repository contains the code and the trained network (`model/best_mean.pt`) used in the paper listed under [Citation](#citation). It covers simulation, reference labelling of the empirical alignments, feature extraction, training and testing.

Results reported in the paper:

| Test set | n | Metric | DNAQFinder | ModelFinder | ModelRevelator |
|---|---:|---|---:|---:|---:|
| Simulated, Seq-Gen (ModelRevelator pipeline) | 1,008 | accuracy | 85.5% | 84.3% | 84.1% |
| TreeBASE (empirical) | 1,144 | agreement with ModelFinder, top-1 / top-2 | 59.5% / 82.1% | reference | 48.6% / 69.9% |
| OrthoMaM + PANDIT internal test (empirical) | 1,920 | agreement with ModelFinder, top-1 / top-2 | 67.5% / 84.6% | reference | 38.1% / 55.3% |

Top-2 counts a prediction as correct when the reference model is one of the two classes with the highest probability. In the paper's CPU runtime test, the average time per alignment of the simulated test set was 12 ms for DNAQFinder and 0.83 s for the model-selection phase of ModelFinder.

## Contents

| Path | Content |
|---|---|
| `dnaqfinder.py` | FASTA reader, per-column counting, network definition, inference |
| `simulation/alisim_yule.py` | AliSim simulation on random Yule–Harding trees |
| `simulation/alisim_mltree.py` | AliSim simulation on maximum-likelihood trees of empirical alignments |
| `simulation/seqgen_modelrevelator.py` | Seq-Gen simulation following the ModelRevelator pipeline |
| `simulation/regime_pool.npz` | sampling pool of (number of sequences, length, p_inv, alpha), see Step 1 |
| `labeling/modelfinder_labels.py` | ModelFinder reference labels for empirical alignments |
| `encoding/encode_counts.py` | FASTA → per-column base counts (`.npz`) |
| `training/train.py` | training of the Set Transformer |
| `evaluation/evaluate.py` | accuracy, macro-F1, per-class recall, confusion matrix, top-2 |
| `evaluation/predict.py` | prediction for new FASTA alignments |
| `model/best_mean.pt` | released network, 553,896 parameters; SHA-256 `8fbb2c2050814f25a051a17f2a76d243794d4055cf0d657f0f84876856693aca` |
| `data/seqgen_test_1008.zip` | simulated test set (1,008 alignments, generating model as label) |
| `data/treebase_1144.zip` | TreeBASE test set (1,144 alignments, ModelFinder label) |

## Requirements

- Python 3 with NumPy, pandas and PyTorch (tested with Python 3.13.2, NumPy 2.2.3, pandas 2.2.3 and PyTorch 2.9.0); astroML for the Seq-Gen simulation.

  ```bash
  pip install -r requirements.txt
  ```

- External programs, needed only for steps 1–2:
  - IQ-TREE 2, used for AliSim and ModelFinder. The paper used 2.3.6 for simulation, and 2.4.0 and 2.3.6 for the labels.
  - Seq-Gen 1.3.5.
  - The files `trees.csv.gz`, `parameters_lanfear.tsv.gz` and `branchlen_lanfear.tsv.gz` from `simulation_parameters/` of [ModelRevelator](https://github.com/Cibiv/ModelRevelator) (MIT licence). They are not redistributed here.
- A GPU for training. Evaluation and prediction run on a CPU.

All commands below are run from the repository root. On Windows, keep the paths passed to IQ-TREE free of non-ASCII characters.

## Reproducing the test results

```bash
python evaluation/evaluate.py
```

The script extracts the two bundled test sets, predicts with `model/best_mean.pt` and compares the result with the values in the paper. The network has no stochastic component at inference, and the column subsample is deterministic. The output is therefore the same on any CPU:

| Test set | Metric | Value |
|---|---|---:|
| Simulated (1,008) | accuracy | 0.8552 |
| | macro-F1 | 0.8539 |
| | top-2 | 0.9613 |
| | recall JC / K2P / F81 / HKY / TN93 / GTR | 0.982 / 0.964 / 0.869 / 0.833 / 0.673 / 0.810 |
| TreeBASE (1,144) | agreement with ModelFinder | 0.5953 |
| | macro-F1 | 0.5089 |
| | top-2 | 0.8208 |
| | recall JC / K2P / F81 / HKY / TN93 / GTR | 0.184 / 0.564 / 0.222 / 0.579 / 0.598 / 0.722 |

The full run took 54 s on a laptop (AMD Ryzen 7 8845HS, default PyTorch threading).

## Pipeline

```text
1. simulation   AliSim (Yule-Harding trees, ML trees) + Seq-Gen  ->  FASTA + ground_truth.csv
2. labels       ModelFinder on empirical alignments              ->  labels.csv (+ ML trees for step 1)
3. extraction   FASTA -> per-column counts (.npz); 17 features per column inside the network
4. training     Set Transformer on simulated + empirical alignments
5. testing      evaluate.py / predict.py
```

Steps 1 and 2 are independent, except that `alisim_mltree.py` needs the ML trees produced in step 2 for the empirical training alignments.

### Step 1. Simulation

All simulated sets are balanced across the six classes. The label of an alignment is the base substitution model used to generate it. The AliSim scripts write this label as `TN` for TN93, the Seq-Gen script as `TN93`.

**AliSim, Yule–Harding trees** (`simulation/alisim_yule.py`). Every alignment has `+I+G4` rate heterogeneity. The parameters are drawn as follows:

| Parameter | Distribution |
|---|---|
| number of sequences, length, p_inv, alpha | one tuple drawn jointly from `regime_pool.npz`: 8–100 sequences; 500, 1,000, 1,500, 2,000 or 3,000 sites; p_inv in [0, 0.5]; alpha in [0.1, 5] |
| κ (K2P, HKY); both TN93 rates | log-uniform on [1, 15] |
| six GTR exchangeabilities | log-uniform on [0.3, 8] |
| base frequencies (F81, HKY, TN93, GTR) | Dirichlet(a, a, a, a) with a ~ U(0.5, 6), clipped at 0.05 and renormalised |
| tree | AliSim `RANDOM{yh/n}` |
| branch-length scale | log-uniform on [0.02, 0.6] |

`regime_pool.npz` contains 100,000 tuples. They were parsed from the file names of an earlier simulated training set of 887,307 alignments, whose sequences were not kept.

**AliSim, ML trees** (`simulation/alisim_mltree.py`). The substitution parameters are drawn as above. The tree is an IQ-TREE ML tree of a randomly chosen empirical training alignment; it fixes the topology, the number of sequences and the branch lengths. The branch lengths are multiplied by a factor drawn log-uniformly from [0.7, 1.4]. Length, p_inv and alpha come from the regime pool.

**Seq-Gen, ModelRevelator pipeline** (`simulation/seqgen_modelrevelator.py`). This script re-implements the simulation in ModelRevelator's `simulate_summarystats.py`:

- Exchangeabilities and base frequencies are sampled from the empirical Lanfear distributions.
- Trees with 8, 16, 64 or 128 sequences come from ModelRevelator's tree file. Their branch lengths are resampled from the Lanfear distributions.
- Every alignment has 1,000 sites.
- Half of the alignments have `+G`, with alpha cycling through 17 values from 0.001 to 10. No alignment has `+I`.

Commands for the training sets of the released model:

```bash
# AliSim, Yule-Harding trees: 18,000 + 90,000 = 108,000 alignments
python simulation/alisim_yule.py --n 18000 --seed 7 --out sim/alisim_train --iqtree <iqtree2>
python simulation/alisim_yule.py --n 90000 --seed <seed> --out sim/alisim_train2 --iqtree <iqtree2>

# AliSim, ML trees of the empirical training alignments (from step 2): 60,000 alignments
python simulation/alisim_mltree.py --n 60000 --seed 11 --tree-dir labels/train/iqtree \
    --out sim/alisim_train_rt --iqtree <iqtree2>

# Seq-Gen: 18,000 + 3 x 12,000 = 54,000 alignments
python simulation/seqgen_modelrevelator.py --n 18000 --seed 20260701 \
    --params-dir <ModelRevelator>/simulation_parameters --seqgen <seq-gen> --out sim/mr_sim_train
python simulation/seqgen_modelrevelator.py --n 12000 --seed <seed> ... --out sim/mr_sim_train_b
python simulation/seqgen_modelrevelator.py --n 12000 --seed <seed> ... --out sim/mr_sim_train_c
python simulation/seqgen_modelrevelator.py --n 12000 --seed <seed> ... --out sim/mr_sim_train_d

# Seq-Gen test set: 1,008 alignments (bundled as data/seqgen_test_1008.zip)
python simulation/seqgen_modelrevelator.py --n 1008 --seed 20260625 ... --out sim/seqgen_test_1008
```

The seeds of `alisim_train2` and `mr_sim_train_b/c/d` were not recorded. Any seed different from those of the other sets draws from the same distributions. Simulated validation sets were also generated, but they were not used to select the released checkpoint. Each Seq-Gen run numbers its alignments from `sim0000`, so write every run to its own folder.

### Step 2. Empirical alignments and reference labels

| Split | OrthoMaM | PANDIT | TreeBASE | Total |
|---|---:|---:|---:|---:|
| Training | 8,000 | 7,343 | 0 | 15,343 |
| Validation | 1,000 | 856 | 0 | 1,856 |
| Internal test | 960 | 960 | 0 | 1,920 |
| External test | 0 | 0 | 1,144 | 1,144 |

The internal test set consists of 200 alignments (100 per source) from the original split, plus 1,720 alignments added later from OrthoMaM v12 and PANDIT 17.0. The added genes and families do not occur in the training, validation or original internal-test data. TreeBASE is used only for testing.

Empirical alignments have no known generating model. The reference label is the model selected by ModelFinder under BIC among the six matrices, each with `+I+G4`:

```bash
python labeling/modelfinder_labels.py --fasta-dir empirical/train --out labels/train --iqtree <iqtree2>
```

For each alignment the script runs

```bash
iqtree2 -s <aln> -m MF -mset JC,K2P,F81,HKY,TN,GTR -mrate I+G4 -keep-ident -T 1 -pre <prefix> -mfreq ""
```

and maps the first token of `Best-fit model according to BIC` to the six classes, for example `GTR+F+I+G4` → GTR. The IQ-TREE outputs are kept in `labels/<split>/iqtree/`; the `.treefile` files of the training split are the input trees of `alisim_mltree.py`.

In the paper, the labels of the training set, validation set, original internal test set and TreeBASE were produced with IQ-TREE 2.4.0, and those of the 1,720 added internal-test alignments with IQ-TREE 2.3.6. In all 20,263 outputs, every F81, HKY, TN and GTR model carries `+F`, and every BIC table lists exactly 6 models. Some IQ-TREE versions also evaluate equal-frequency variants of F81, HKY, TN and GTR and print them without `+F` (these are JC, K2P, TNe and SYM). The script flags such rows in the column `freq_check`. ModelFinder failed on 8 of the 1,152 TreeBASE alignments, so 1,144 remain.

### Step 3. Feature extraction

Each alignment is converted to a matrix of per-column counts of A, C, G, T and other characters (gap, N, IUPAC codes). Lower-case bases are counted as upper case, and sequences are truncated to the shortest one.

```bash
# simulated sets: one .npz per folder, labels from ground_truth.csv
python encoding/encode_counts.py --fasta-dir sim/alisim_train/fasta \
    --labels sim/alisim_train/ground_truth.csv --out encoded/alisim_train.npz
# empirical sets: labels and paths from labels.csv of step 2
python encoding/encode_counts.py --fasta-dir empirical/train --labels labels/train/labels.csv \
    --id-col file --label-col label --path-col file --out encoded/emp_train.npz
```

`train.py` expects the files `mr_sim_train.npz`, `mr_sim_train_b.npz`, `mr_sim_train_c.npz`, `mr_sim_train_d.npz`, `alisim_train.npz`, `alisim_train2.npz`, `alisim_train_rt.npz`, `emp_train.npz` and `emp_val.npz`. Other names can be passed with `--sim-train`, `--emp-train` and `--val`.

The network computes 17 features for every column from the counts c = (c_A, c_C, c_G, c_T), n = c_A + c_C + c_G + c_T and the count g of other characters (`dnaqfinder.FrontEnd`):

| # | Feature | Definition |
|---|---|---|
| 1–4 | base composition | c_k / n |
| 5 | gap fraction | g / (n + g) |
| 6–15 | pair frequencies AA, CC, GG, TT, AC, AG, AT, CG, CT, GT | (c_i c_j − [i = j] c_i) / (n (n − 1)) |
| 16 | entropy | −Σ_k (c_k / n) ln (c_k / n) |
| 17 | depth | ln n |

AG and CT are the transition pairs. The features do not depend on the order of the sequences. The network has no positional encoding, so for alignments of at most 1,500 columns the prediction does not depend on the order of the columns either. Long alignments are subsampled:
- during training, 800 columns are drawn at random without replacement per example and epoch;
- at inference, 1,500 evenly spaced columns are used.

### Step 4. Training

```bash
python training/train.py --data-dir encoded --out runs/m_p3
```

The defaults are the settings of the released model:

| Item | Setting |
|---|---|
| network | linear 17 → 128; 2 ISAB blocks (16 inducing points, 4 heads, feed-forward 128–256–128, GELU, dropout 0.1); masked mean over columns; LayerNorm, linear 128 → 128, GELU, dropout 0.1, linear 128 → 6; 553,896 parameters |
| training data | Seq-Gen 54,000 + AliSim Yule–Harding 108,000 + AliSim ML trees 60,000 + empirical 15,343 × 4 = 283,372 examples per epoch (empirical alignments make up 21.7%) |
| validation | 1,856 empirical alignments; the checkpoint with the highest validation accuracy is kept (0.7112 for the released model) |
| optimiser | AdamW, learning rate 3·10⁻⁴, weight decay 10⁻⁴, batch size 128 |
| schedule | 30 epochs; linear warm-up over 3 epochs, then cosine decay, updated once per epoch |
| loss | cross-entropy with label smoothing 0.05; gradient norm clipped at 1.0 |
| initialisation | `torch.manual_seed(0)`; column sampling uses `numpy.random.default_rng(0)` |

The model is trained once from random initialisation on simulated and empirical alignments mixed in the same batches. No pre-training or fine-tuning is involved. The released checkpoint was trained on a Kaggle GPU. On Windows, pass `--workers 0`.

### Step 5. Testing and prediction

```bash
# bundled test sets, compared with the paper
python evaluation/evaluate.py
# any labelled set
python evaluation/evaluate.py --fasta-dir <folder> --labels <csv> --id-col sample_id --label-col true_label
# a checkpoint from step 4
python evaluation/evaluate.py --ckpt runs/m_p3/best_mean.pt
# new alignments
python evaluation/predict.py <fasta file or folder> -o predictions.csv
```

`evaluate.py` reports accuracy, macro-F1, top-2, per-class recall and the confusion matrix. For empirical sets, "accuracy" is the agreement with ModelFinder. The internal test set (1,920 alignments) is not bundled.

## Reproducibility notes

- **Simulators are not seeded.** The scripts seed the parameter draws, but they do not pass a seed to AliSim or Seq-Gen. A rerun therefore reproduces the model strings, trees and branch-length scales, but not the sequences. The released network was trained on the original simulated alignments. These and the empirical training, validation and internal-test alignments (about 17 GB in total) are not part of this repository; they are available from the authors on request.
- **Training is not bit-reproducible.** GPU kernels and the order of the training files change the result. In addition, every DataLoader worker starts from the same column-sampling generator, so the sampled columns depend on `--workers`. The released checkpoint is the one evaluated in the paper.

## Citation

Le Si Thai Son, Phan Cong Duong, Tran Ho Manh Duc, Nguyen Huy Tinh, Le Sy Vinh. DNAQFinder: a fast machine learning method for nucleotide substitution model selection on empirical alignments. Accepted at RIVF 2026.

## References

- Burgstaller-Muehlbacher S, Crotty SM, Schmidt HA, Reden F, Drucks T, von Haeseler A (2023). ModelRevelator: Fast phylogenetic model estimation via deep learning. *Molecular Phylogenetics and Evolution* 188. doi:10.1016/j.ympev.2023.107905
- Kalyaanamoorthy S, Minh BQ, Wong TKF, von Haeseler A, Jermiin LS (2017). ModelFinder: Fast model selection for accurate phylogenetic estimates. *Nature Methods* 14(6):587–589. doi:10.1038/nmeth.4285
- Lee J, Lee Y, Kim J, Kosiorek AR, Choi S, Teh YW (2019). Set Transformer: A framework for attention-based permutation-invariant neural networks. *Proceedings of the 36th International Conference on Machine Learning*, PMLR 97:3744–3753.
- Ly-Trong N, Naser-Khdour S, Lanfear R, Minh BQ (2022). AliSim: A fast and versatile phylogenetic sequence simulator for the genomic era. *Molecular Biology and Evolution* 39(5):msac092. doi:10.1093/molbev/msac092
- Minh BQ, Schmidt HA, Chernomor O, Schrempf D, Woodhams MD, von Haeseler A, Lanfear R (2020). IQ-TREE 2: New models and efficient methods for phylogenetic inference in the genomic era. *Molecular Biology and Evolution* 37(5):1530–1534. doi:10.1093/molbev/msaa015
- Rambaut A, Grassly NC (1997). Seq-Gen: an application for the Monte Carlo simulation of DNA sequence evolution along phylogenetic trees. *Bioinformatics* 13(3):235–238. doi:10.1093/bioinformatics/13.3.235
