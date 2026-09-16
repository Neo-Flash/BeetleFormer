"""Dataset, tokenizer and gene-disjoint splitting for RNAi activity prediction.

Input  : a dsRNA nucleotide sequence (A/C/G/T), single-nucleotide tokenized.
Targets:
  * lethality  (N, 3) in [0,1]  with a (N, 3) observed-mask  [main, regression]
  * phenotype  (N, 15) multi-hot {0,1}                       [aux, multi-label]

The manifest is produced by scripts/prepare_data.py. Sequences vary 130-615 bp;
we pad/truncate to `max_len` and return an attention mask so the encoder ignores
padding. Lethality percentages (0-100) are scaled to [0,1]; missing entries are
masked out of the loss and metrics.
"""
import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import Dataset

# single-nucleotide vocabulary; 0 = PAD, then ACGT, N for unknown
PAD, VOCAB = 0, {"A": 1, "C": 2, "G": 3, "T": 4, "U": 4, "N": 5}
VOCAB_SIZE = 6
# Main lethality target. dali11 = lethality 11 days after larval injection, the
# densest and most standard iBeetle readout. dali22 (larval day-22) is only
# measured in screening phase I and dapi11 (pupal injection) is a different assay,
# so both are excluded from the main task; add them back here to train jointly.
LETH_COLS = ["dali11"]


def _kmer_index(k):
    """Build (and cache) a k-mer -> id table. id 0 = PAD, 4**k+1 = unknown."""
    if k == 1:
        return None
    cache = _kmer_index.__dict__.setdefault("_c", {})
    if k not in cache:
        from itertools import product
        idx = {"".join(p): i + 1 for i, p in enumerate(product("ACGT", repeat=k))}
        cache[k] = idx
    return cache[k]


def kmer_vocab_size(k):
    return VOCAB_SIZE if k == 1 else 4 ** k + 2  # +PAD +unknown


def encode_seq(seq, max_len, k=1):
    """Sequence string -> (token_ids, attention_mask) at k-mer resolution.

    k=1 uses the single-nucleotide vocabulary; k>1 tokenizes into non-overlapping
    k-mers (stride k). U is mapped to T; any k-mer containing an unknown base maps
    to the unknown id. Padded/truncated to max_len // k tokens.
    """
    s = seq[: max_len].upper().replace("U", "T")
    if k == 1:
        ids = [VOCAB.get(c, VOCAB["N"]) for c in s]
        n_tok = max_len
    else:
        table = _kmer_index(k)
        unk = 4 ** k + 1
        ids = [table.get(s[i:i + k], unk) for i in range(0, len(s) - k + 1, k)]
        n_tok = max_len // k
    n = len(ids)
    if n < n_tok:
        ids = ids + [PAD] * (n_tok - n)
    else:
        ids = ids[:n_tok]; n = n_tok
    mask = [1] * min(n, n_tok) + [0] * max(0, n_tok - n)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def _read_manifest(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_topics(data_dir):
    with open(os.path.join(data_dir, "topics.txt")) as f:
        return [t for t in f.read().splitlines() if t]


class RNAiDataset(Dataset):
    """One sample per dsRNA: tokenized sequence + lethality (+mask) + phenotype."""

    def __init__(self, manifest_csv, topics, max_len=640, rows=None, kmer=1,
                 use_features=False):
        self.rows = rows if rows is not None else _read_manifest(manifest_csv)
        self.topics = topics
        self.max_len = max_len
        self.kmer = kmer
        self.use_features = use_features

    def __len__(self):
        return len(self.rows)

    def _lethality(self, row):
        vals = np.zeros(len(LETH_COLS), dtype=np.float32)
        mask = np.zeros(len(LETH_COLS), dtype=bool)
        for i, c in enumerate(LETH_COLS):
            v = row.get(c, "")
            if v not in ("", "NA"):
                vals[i] = float(v) / 100.0   # 0-100% -> [0,1]
                mask[i] = True
        return torch.from_numpy(vals), torch.from_numpy(mask)

    def __getitem__(self, idx):
        row = self.rows[idx]
        ids, attn = encode_seq(row["seq"], self.max_len, self.kmer)
        leth, leth_mask = self._lethality(row)
        pheno = torch.tensor([float(row.get(t, 0)) for t in self.topics],
                             dtype=torch.float32)
        sample = {"ids": ids, "attn": attn, "lethality": leth,
                  "lethality_mask": leth_mask, "phenotype": pheno,
                  "tc": row.get("tc", ""), "iB": row.get("iB", "")}
        if self.use_features:
            from data.features import compute_features
            sample["feat"] = torch.from_numpy(
                compute_features(row["seq"], self.max_len))
        return sample


def collate(batch):
    out = {
        "ids": torch.stack([b["ids"] for b in batch]),
        "attn": torch.stack([b["attn"] for b in batch]),
        "lethality": torch.stack([b["lethality"] for b in batch]),
        "lethality_mask": torch.stack([b["lethality_mask"] for b in batch]),
        "phenotype": torch.stack([b["phenotype"] for b in batch]),
        "tc": [b["tc"] for b in batch],
        "iB": [b["iB"] for b in batch],
    }
    if "feat" in batch[0]:
        out["feat"] = torch.stack([b["feat"] for b in batch])
    return out


def gene_disjoint_split(rows, val_fraction=0.1, seed=508):
    """Split rows so that no gene (tc) appears in both train and val.

    dsRNAs without a gene id are treated as their own singleton group (kept
    together by iB so they never leak a gene across the split).
    """
    by_gene = {}
    for i, r in enumerate(rows):
        key = r.get("tc") or f"_solo_{r.get('iB', i)}"
        by_gene.setdefault(key, []).append(i)
    genes = sorted(by_gene.keys())
    rng = random.Random(seed)
    rng.shuffle(genes)

    n_val_target = int(len(rows) * val_fraction)
    val_idx, train_idx, n_val = [], [], 0
    for g in genes:
        if n_val < n_val_target:
            val_idx.extend(by_gene[g])
            n_val += len(by_gene[g])
        else:
            train_idx.extend(by_gene[g])
    return sorted(train_idx), sorted(val_idx)
