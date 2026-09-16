"""Per-dsRNA physicochemical / thermodynamic and siRNA-tiling features.

These features complement the learned sequence encoder with hand-crafted signal
that recurs across the RNAi-efficacy literature (thermodynamic stability, base
composition, and empirical siRNA design rules). Everything here is a pure-python
implementation of published scientific facts, not third-party code:

  * nearest-neighbour stacking free energies (Turner-style RNA parameters) for a
    coarse duplex-stability proxy;
  * Wallace-rule melting temperature;
  * sliding-window and terminal GC content;
  * dinucleotide frequencies;
  * Reynolds et al. (2004) empirical siRNA design rules, tiled over the dsRNA to
    model a long dsRNA as a bag of constituent siRNAs.

References (methodology only, implemented independently):
  Reynolds et al., Nat. Biotechnol. 22 (2004) 326-330 (siRNA design rules);
  OligoFormer / siRNADiscovery / DeepSilencer (thermodynamic feature use).
"""
import numpy as np

# Coarse nearest-neighbour stacking free energies (kcal/mol), RNA-like, keyed by
# the 5'->3' dinucleotide. Values are a compact approximation; only relative
# magnitude matters as a learned-model input feature.
_NN_DG = {
    "AA": -0.9, "AU": -1.1, "UA": -1.3, "UU": -0.9,
    "AC": -2.2, "CA": -2.1, "GU": -2.1, "UG": -1.4,
    "AG": -2.1, "GA": -2.4, "CU": -2.1, "UC": -2.2,
    "CC": -3.3, "GG": -3.3, "CG": -2.4, "GC": -3.4,
}
_BASES = "ACGT"
_DINUC = [a + b for a in _BASES for b in _BASES]  # 16, fixed order


def _clean(seq):
    return seq.upper().replace("U", "T")


def _gc(s):
    if not s:
        return 0.0
    return (s.count("G") + s.count("C")) / len(s)


def _nn_dg(seq):
    """Sum of nearest-neighbour stacking free energies, normalised by length."""
    s = seq.replace("T", "U")
    if len(s) < 2:
        return 0.0
    tot = sum(_NN_DG.get(s[i:i + 2], -2.0) for i in range(len(s) - 1))
    return tot / (len(s) - 1)


def _tm_wallace(seq):
    """Wallace-rule melting temperature: 2(A+T) + 4(G+C), normalised."""
    s = seq
    at = s.count("A") + s.count("T")
    gc = s.count("G") + s.count("C")
    return (2 * at + 4 * gc) / max(1, len(s))


def _reynolds_score(w):
    """Reynolds et al. (2004) design-rule score for a 19-mer window (0-6).

    Uses the position-independent and position-based criteria that can be read
    off a single strand: GC 30-52%, >=3 A/T in positions 15-19, A at pos 3,
    U(=T) at pos 10, A at pos 19, and a non-G/C base at pos 19.
    """
    if len(w) < 19:
        return 0.0
    score = 0
    gc = _gc(w[:19])
    if 0.30 <= gc <= 0.52:
        score += 1
    if sum(1 for c in w[14:19] if c in "AT") >= 3:
        score += 1
    if w[2] == "A":
        score += 1
    if w[9] == "T":
        score += 1
    if w[18] == "A":
        score += 1
    if w[18] not in "GC":
        score += 1
    return score / 6.0


def _tiling(seq, k=19):
    """Aggregate Reynolds scores over all k-mer windows: mean and good-fraction."""
    if len(seq) < k:
        return 0.0, 0.0
    scores = [_reynolds_score(seq[i:i + k]) for i in range(len(seq) - k + 1)]
    scores = np.asarray(scores)
    return float(scores.mean()), float(np.mean(scores >= 0.5))


# ---------------------------------------------------------------------------- #
# feature names, for reproducibility / paper tables
FEATURE_NAMES = (
    ["frac_A", "frac_C", "frac_G", "frac_T", "gc", "len_norm",
     "nn_dg", "tm", "gc_win_mean", "gc_win_std", "gc_5p", "gc_3p",
     "tile_reynolds_mean", "tile_reynolds_goodfrac"]
    + [f"di_{d}" for d in _DINUC]
)
FEATURE_DIM = len(FEATURE_NAMES)  # 14 + 16 = 30


def compute_features(seq, max_len=640, win=50):
    """Return a fixed FEATURE_DIM vector of features for one dsRNA sequence."""
    s = _clean(seq)
    n = len(s)
    feats = []
    # base composition
    for b in _BASES:
        feats.append(s.count(b) / max(1, n))
    feats.append(_gc(s))                       # gc
    feats.append(n / max_len)                  # length normalised
    feats.append(_nn_dg(s))                    # duplex-stability proxy
    feats.append(_tm_wallace(s))               # Tm
    # sliding-window GC statistics
    if n >= win:
        gcs = [_gc(s[i:i + win]) for i in range(0, n - win + 1, win)]
        feats.append(float(np.mean(gcs)))
        feats.append(float(np.std(gcs)))
    else:
        feats.append(_gc(s)); feats.append(0.0)
    # terminal GC (first / last 50 nt)
    feats.append(_gc(s[:win]))
    feats.append(_gc(s[-win:]))
    # Reynolds tiling
    tmean, tgood = _tiling(s)
    feats.append(tmean); feats.append(tgood)
    # dinucleotide frequencies (16)
    dc = {d: 0 for d in _DINUC}
    for i in range(n - 1):
        d = s[i:i + 2]
        if d in dc:
            dc[d] += 1
    denom = max(1, n - 1)
    feats.extend(dc[d] / denom for d in _DINUC)
    return np.asarray(feats, dtype=np.float32)
