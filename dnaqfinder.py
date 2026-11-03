"""DNAQFinder: shared code for encoding, the network and inference.

Pipeline (no tree reconstruction, no likelihood computation):
    FASTA -> per-column base counts [L, 5] (A, C, G, T, other)
          -> FrontEnd: 17 features per column
          -> Set Transformer (2 ISAB blocks, masked mean pooling)
          -> 6 logits (JC, K2P, F81, HKY, TN, GTR)

TN denotes TN93 throughout. The released checkpoint is model/best_mean.pt.
"""
import os

import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------------
CLASSES = ['JC', 'K2P', 'F81', 'HKY', 'TN', 'GTR']
CIDX = {c: i for i, c in enumerate(CLASSES)}
_ALIAS = {'JC69': 'JC', 'K80': 'K2P', 'K2P': 'K2P', 'TN93': 'TN', 'TNE': 'TN',
          'HKY85': 'HKY', 'HKY': 'HKY', 'F81': 'F81', 'JC': 'JC', 'GTR': 'GTR'}


def canon(model):
    """Map a model string such as 'HKY+F+I+G4' or 'TN93' to one of CLASSES."""
    m = str(model).strip().upper().split('+')[0]
    return _ALIAS.get(m, m)


def cidx(model):
    """Model string -> class index 0..5, or -1 if it is not one of the 6 classes."""
    return CIDX.get(canon(model), -1)


# ----------------------------------------------------------------------------
# FASTA -> per-column counts
# ----------------------------------------------------------------------------
# A/a=0, C/c=1, G/g=2, T/t=3; every other byte (gap, N, IUPAC code, '?') = 4
_LUT = np.full(256, 4, np.uint8)
for _ch, _k in [('A', 0), ('C', 1), ('G', 2), ('T', 3)]:
    _LUT[ord(_ch)] = _k
    _LUT[ord(_ch.lower())] = _k


def read_fasta_codes(path):
    """Read a FASTA alignment -> uint8 matrix [n_sequences, L] of codes 0..4.

    Sequences are truncated to the shortest one. Returns None if the file has
    fewer than 2 sequences or zero length."""
    seqs, cur = [], []
    with open(path, 'rb') as fh:
        for ln in fh:
            if ln[:1] == b'>':
                if cur:
                    seqs.append(b''.join(cur)); cur = []
            else:
                cur.append(ln.strip())
    if cur:
        seqs.append(b''.join(cur))
    if len(seqs) < 2:
        return None
    L = min(len(s) for s in seqs)
    if L == 0:
        return None
    mat = np.empty((len(seqs), L), np.uint8)
    for i, s in enumerate(seqs):
        mat[i] = _LUT[np.frombuffer(s[:L], np.uint8)]
    return mat


def col_counts(mat):
    """[n_sequences, L] codes -> int16 [L, 5] counts of (A, C, G, T, other) per column."""
    L = mat.shape[1]
    out = np.zeros((L, 5), np.int16)
    for k in range(5):
        out[:, k] = (mat == k).sum(0)
    return out


def encode_fasta(path):
    """FASTA path -> per-column counts [L, 5] (int16), or None if unreadable."""
    mat = read_fasta_codes(path)
    if mat is None:
        return None
    return col_counts(mat)


def load_npz(path):
    """Load a packed .npz written by encoding/encode_counts.py.

    Returns (ids[N], labels[N], list of N count arrays [L_i, 5])."""
    d = np.load(path, allow_pickle=True)
    counts, off, labels = d['counts'], d['offsets'], d['labels']
    ids = d['ids'] if 'ids' in d.files else np.array([str(i) for i in range(len(labels))])
    lst = [counts[off[i]:off[i + 1]] for i in range(len(labels))]
    return np.asarray([str(x) for x in ids]), np.asarray(labels, int), lst


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------
PAIR_IDX = [(0, 0), (1, 1), (2, 2), (3, 3), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
D, HEADS, M, N_ISAB = 128, 4, 16, 2
EVAL_MAX_COLS = 1500   # columns used at inference (evenly spaced subsample if L > 1500)


class FrontEnd(nn.Module):
    """Per-column counts [..., 5] -> 17 features per column, then LayerNorm.

    With c = (c_A, c_C, c_G, c_T), n = sum(c) and g = count of other characters:
      1-4   base composition      c_k / n
      5     gap/other fraction    g / (n + g)
      6-15  pair frequencies      (c_i c_j - [i=j] c_i) / (n (n - 1)) for
                                  AA, CC, GG, TT, AC, AG, AT, CG, CT, GT
      16    Shannon entropy of the composition
      17    log n
    """
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(17)

    def forward(self, counts):
        c = counts[..., :4]; g = counts[..., 4:5]
        nacgt = c.sum(-1, keepdim=True).clamp(min=1.); total = (nacgt + g).clamp(min=1.)
        comp = c / nacgt; gapf = g / total
        Mm = c.unsqueeze(-1) * c.unsqueeze(-2) - torch.diag_embed(c)
        Mn = Mm / (nacgt * (nacgt - 1.)).clamp(min=1.).unsqueeze(-1)
        pair = torch.stack([Mn[..., i, j] for i, j in PAIR_IDX], -1)
        ent = -(comp.clamp(min=1e-8) * comp.clamp(min=1e-8).log()).sum(-1, keepdim=True)
        return self.ln(torch.cat([comp, gapf, pair, ent, nacgt.log()], -1))


class MAB(nn.Module):
    """Multihead attention block: LN(Q + MHA(Q, K, K)), then LN(H + FFN(H))."""
    def __init__(self, d, h, fm=2, p=.1):
        super().__init__()
        self.a = nn.MultiheadAttention(d, h, p, batch_first=True)
        self.l1 = nn.LayerNorm(d); self.l2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * fm), nn.GELU(), nn.Linear(d * fm, d))

    def forward(self, Q, K, kpm=None):
        a, _ = self.a(Q, K, K, key_padding_mask=kpm, need_weights=False)
        H = self.l1(Q + a)
        return self.l2(H + self.ff(H))


class ISAB(nn.Module):
    """Induced set attention block with m inducing points."""
    def __init__(self, d, h, m):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m, d) * .1); self.m1 = MAB(d, h); self.m2 = MAB(d, h)

    def forward(self, X, kpm=None):
        H = self.m1(self.I.expand(X.size(0), -1, -1), X, kpm)
        return self.m2(X, H)


class PMA(nn.Module):
    """Pooling by multihead attention (used only by the pool='attn' variant)."""
    def __init__(self, d, h, k=1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, k, d) * .1); self.m = MAB(d, h)

    def forward(self, X, kpm=None):
        return self.m(self.S.expand(X.size(0), -1, -1), X, kpm)


class SetTransformer(nn.Module):
    """17 -> 128 projection, 2 ISAB blocks over columns, pooling, classification head.

    pool='mean' (masked mean) is the released model (553,896 parameters);
    pool='attn' replaces the mean with PMA (686,504 parameters)."""
    def __init__(self, pool='mean'):
        super().__init__()
        self.front = FrontEnd(); self.proj = nn.Linear(17, D); self.pool = pool
        self.isabs = nn.ModuleList([ISAB(D, HEADS, M) for _ in range(N_ISAB)])
        if pool == 'attn':
            self.pma = PMA(D, HEADS, 1)
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(),
                                  nn.Dropout(.1), nn.Linear(D, 6))

    def forward(self, counts, mask):
        kpm = ~mask
        x = self.proj(self.front(counts))
        for isab in self.isabs:
            x = isab(x, kpm)
        if self.pool == 'attn':
            z = self.pma(x, kpm).squeeze(1)
        else:
            w = mask.unsqueeze(-1).float()
            z = (x * w).sum(1) / w.sum(1).clamp(min=1.)
        return self.head(z)


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(REPO_ROOT, 'model', 'best_mean.pt')


def load_model(ckpt=DEFAULT_CKPT, pool='mean'):
    """Load a checkpoint into a SetTransformer in eval mode (CPU)."""
    model = SetTransformer(pool)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()
    return model


@torch.no_grad()
def predict_counts(model, counts_list, bs=64):
    """Predict for a list of per-column count arrays ([L_i, 5] each).

    Columns are capped at EVAL_MAX_COLS by a deterministic evenly spaced
    subsample, then zero-padded per batch with a key-padding mask, so a
    prediction does not depend on the other alignments in the batch.

    Returns (pred[N], probs[N, 6], logits[N, 6], mean forward time per alignment in s)."""
    import time
    N = len(counts_list)
    pred = np.zeros(N, int); probs = np.zeros((N, 6), np.float32); logits = np.zeros((N, 6), np.float32)
    t = 0.0
    for st in range(0, N, bs):
        seqs = []
        for i in range(st, min(st + bs, N)):
            cc = np.asarray(counts_list[i], np.float32); L = cc.shape[0]
            if L > EVAL_MAX_COLS:
                cc = cc[np.sort(np.linspace(0, L - 1, EVAL_MAX_COLS).astype(int))]
            seqs.append(torch.from_numpy(cc))
        Lmax = max(x.shape[0] for x in seqs); B = len(seqs)
        x = torch.zeros(B, Lmax, 5); mask = torch.zeros(B, Lmax, dtype=torch.bool)
        for j, sq in enumerate(seqs):
            x[j, :sq.shape[0]] = sq; mask[j, :sq.shape[0]] = True
        t0 = time.perf_counter()
        out = model(x, mask)
        t += time.perf_counter() - t0
        p = torch.softmax(out, dim=1)
        logits[st:st + B] = out.numpy(); probs[st:st + B] = p.numpy()
        pred[st:st + B] = out.argmax(1).numpy()
    return pred, probs, logits, t / max(N, 1)
