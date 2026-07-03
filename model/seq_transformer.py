"""dsRNA sequence Transformer for RNAi activity prediction.

A from-scratch encoder over single-nucleotide tokens, with two heads:
  * lethality head   -> len(LETH_COLS) outputs (main: dali11), sigmoid to [0,1]
  * phenotype head   -> 15 logits (multi-label, BCE-with-logits externally)

An optional convolutional stem (``conv_stem=True``) applies a small stack of
depthwise-style Conv1d layers over the token embeddings before the Transformer,
to give the model a local-motif inductive bias and to downsample the 130-615 bp
sequence so self-attention runs over a shorter length. This mirrors the
conv-front-end + sequence-model design shared by dsRNAPredictor, OligoFormer and
siRNADiscovery, and is the recommended configuration for long dsRNA. Without the
stem the model is a plain single-nucleotide Transformer.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.rnai_dataset import VOCAB_SIZE, PAD, LETH_COLS


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class ConvStem(nn.Module):
    """Local-motif front end: a stack of Conv1d+GELU over the embedding, with a
    final max-pool that downsamples the sequence (and its mask) by ``pool``."""

    def __init__(self, dim, kernel=9, layers=2, pool=4, dropout=0.1):
        super().__init__()
        blocks = []
        for _ in range(layers):
            blocks += [
                nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.conv = nn.Sequential(*blocks)
        self.pool = pool

    def forward(self, x, attn):
        """x,(B,L,dim) attn,(B,L) bool. Returns (x',attn') downsampled by pool."""
        h = self.conv(x.transpose(1, 2))                     # (B,dim,L)
        if self.pool > 1:
            h = F.max_pool1d(h, kernel_size=self.pool, ceil_mode=True)
            m = F.max_pool1d(attn.float().unsqueeze(1), kernel_size=self.pool,
                             ceil_mode=True).squeeze(1)       # (B,L') real if any real
            attn = m > 0.5
        return h.transpose(1, 2), attn                       # (B,L',dim), (B,L')


class dsRNATransformer(nn.Module):
    def __init__(self, dim=192, depth=4, heads=6, mlp_ratio=4.0, dropout=0.1,
                 max_len=640, num_lethality=len(LETH_COLS), num_topics=15,
                 pool="mean", conv_stem=False, stem_kernel=9, stem_layers=2,
                 stem_pool=4, vocab_size=VOCAB_SIZE, extra_dim=0):
        super().__init__()
        self.dim = dim
        self.pool = pool
        self.extra_dim = extra_dim
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.stem = ConvStem(dim, stem_kernel, stem_layers, stem_pool, dropout) \
            if conv_stem else None
        self.pos = SinusoidalPositionalEncoding(dim, max_len + 1)
        if pool == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.cls, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

        hdim = dim + extra_dim  # concatenated hand-crafted features
        self.lethality_head = nn.Sequential(
            nn.Linear(hdim, dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, num_lethality),
        )
        self.phenotype_head = nn.Sequential(
            nn.Linear(hdim, dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, num_topics),
        )

    def encode(self, ids, attn):
        """ids,(B,L) attn,(B,L) bool True=real. Returns pooled (B,dim)."""
        x = self.embed(ids)
        if self.stem is not None:
            x, attn = self.stem(x, attn)   # local motifs + downsample seq & mask
        key_padding = ~attn  # TransformerEncoder expects True = ignore
        if self.pool == "cls":
            b = x.size(0)
            cls = self.cls.expand(b, -1, -1)
            x = torch.cat([cls, x], dim=1)
            cls_keep = torch.zeros(b, 1, dtype=torch.bool, device=ids.device)
            key_padding = torch.cat([cls_keep, key_padding], dim=1)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=key_padding)
        x = self.norm(x)
        if self.pool == "cls":
            return x[:, 0]
        # masked mean over real tokens
        m = attn.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, ids, attn, feat=None):
        z = self.encode(ids, attn)
        if self.extra_dim and feat is not None:
            z = torch.cat([z, feat], dim=-1)
        return {
            "lethality_logit": self.lethality_head(z),   # raw; sigmoid for [0,1]
            "phenotype_logit": self.phenotype_head(z),   # raw; BCE-with-logits
            "embedding": z,
        }



def masked_regression_loss(pred_logit, target, mask, kind="huber", beta=0.1):
    """Regression loss over observed entries only.

    pred_logit: (B,3) raw; sigmoid -> [0,1] to match scaled targets.
    target:     (B,3) in [0,1]; mask: (B,3) bool observed.
    """
    pred = torch.sigmoid(pred_logit)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    diff = (pred - target)[mask]
    if kind == "huber":
        absd = diff.abs()
        quad = torch.clamp(absd, max=beta)
        return ((0.5 * quad ** 2 + beta * (absd - quad)) / beta).mean()
    return (diff ** 2).mean()


def pairwise_ranking_loss(pred_logit, target, mask):
    """RankNet-style pairwise ranking loss on the first lethality readout.

    Directly optimizes ordering (the Spearman objective): for every observed pair
    whose targets differ, penalize a predicted order that disagrees with the true
    order, via a logistic (softplus) surrogate. Column 0 is the headline readout.
    """
    m = mask[:, 0]
    if m.sum() < 2:
        return pred_logit.sum() * 0.0
    p = torch.sigmoid(pred_logit[:, 0][m])   # (K,)
    t = target[:, 0][m]                       # (K,)
    dp = p.unsqueeze(0) - p.unsqueeze(1)      # (K,K) pred_j - pred_i
    dt = t.unsqueeze(0) - t.unsqueeze(1)      # (K,K) true_j - true_i
    sign = torch.sign(dt)                     # +1 if j should rank above i
    valid = sign != 0
    if valid.sum() == 0:
        return pred_logit.sum() * 0.0
    # want sign*dp > 0; logistic surrogate softplus(-sign*dp)
    loss = torch.nn.functional.softplus(-sign[valid] * dp[valid])
    return loss.mean()
