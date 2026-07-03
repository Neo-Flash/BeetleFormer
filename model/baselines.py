"""Baselines for RNAi activity prediction, for fair comparison with the
from-scratch Transformer (model/seq_transformer.py).

  * CNNBiLSTM    - reimplementation of the dsRNAPredictor recipe (Conv1D +
                   BiLSTM + self-attention), but adapted to our multi-task
                   regression+multilabel setting and PyTorch. Same two heads as
                   the Transformer so the comparison isolates the encoder.
  * gc_length_features - trivial hand-feature baseline (GC content, length,
                   per-base composition) for a linear/MLP lower bound.

Both consume the same batch dict as the Transformer (ids, attn, ...).
"""
import torch
import torch.nn as nn

from data.rnai_dataset import VOCAB_SIZE, PAD, LETH_COLS


class CNNBiLSTM(nn.Module):
    """dsRNAPredictor-style encoder: Conv1D -> maxpool -> BiLSTM -> attention
    pool, with our lethality + phenotype heads."""

    def __init__(self, dim=128, num_lethality=len(LETH_COLS), num_topics=15,
                 dropout=0.3, vocab_size=VOCAB_SIZE, extra_dim=0):
        super().__init__()
        self.extra_dim = extra_dim
        self.embed = nn.Embedding(vocab_size, 64, padding_idx=PAD)
        self.conv = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=5, ceil_mode=True)
        self.bn = nn.BatchNorm1d(64)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(64, dim // 2, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(dim, 1)
        hdim = dim + extra_dim
        self.lethality_head = nn.Linear(hdim, num_lethality)
        self.phenotype_head = nn.Linear(hdim, num_topics)
        self.dim = dim

    def encode(self, ids, attn):
        x = self.embed(ids).transpose(1, 2)          # (B,64,L)
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = self.bn(x)
        x = self.drop(x).transpose(1, 2)             # (B,L',64)
        x, _ = self.lstm(x)                          # (B,L',dim)
        w = torch.softmax(self.attn(x).squeeze(-1), dim=1)  # (B,L')
        return torch.bmm(w.unsqueeze(1), x).squeeze(1)      # (B,dim)

    def forward(self, ids, attn, feat=None):
        z = self.encode(ids, attn)
        if self.extra_dim and feat is not None:
            z = torch.cat([z, feat], dim=-1)
        return {
            "lethality_logit": self.lethality_head(z),
            "phenotype_logit": self.phenotype_head(z),
            "embedding": z,
        }


class CNNBiLSTMFaithful(nn.Module):
    """Faithful re-implementation of the dsRNAPredictor network (Cheng et al.),
    with architecture and hyperparameters matched as closely as PyTorch allows.

    Original (Keras): Conv1D(64, k=3, valid, relu) -> MaxPool(5) -> BatchNorm
    -> Dropout(0.2) -> BiLSTM(50, return_sequences) -> BatchNorm -> Dropout(0.5)
    -> SeqSelfAttention(additive, softmax) -> Flatten -> Dense(16, sigmoid)
    -> Dense(1, sigmoid). Trained with Adam lr=1e-4, batch=32.

    The only intentional difference from the original is the input: we use our
    learnable single-nucleotide embedding instead of summed dna2vec k-mer vectors
    (the original pre-trained dna2vec weights are not distributed), so that the
    baseline uses the same input representation as dsRNATransformer and the
    comparison isolates the network architecture.
    """

    def __init__(self, dim=100, num_lethality=len(LETH_COLS), num_topics=15,
                 vocab_size=VOCAB_SIZE, extra_dim=0, lstm_hidden=50,
                 attn_dim=32, dense_hidden=16):
        super().__init__()
        self.extra_dim = extra_dim
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.conv = nn.Conv1d(dim, 64, kernel_size=3, padding=0)   # 'valid'
        self.pool = nn.MaxPool1d(kernel_size=5)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.2)
        self.lstm = nn.LSTM(64, lstm_hidden, batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(2 * lstm_hidden)
        self.drop2 = nn.Dropout(0.5)
        # additive self-attention, matching keras SeqSelfAttention(softmax)
        self.attn_W = nn.Linear(2 * lstm_hidden, attn_dim)
        self.attn_v = nn.Linear(attn_dim, 1)
        self.dense1 = nn.Linear(2 * lstm_hidden, dense_hidden)
        hdim = dense_hidden + extra_dim
        self.lethality_head = nn.Linear(hdim, num_lethality)
        self.phenotype_head = nn.Linear(hdim, num_topics)
        self.dim = dim

    def encode(self, ids, attn):
        x = self.embed(ids).transpose(1, 2)          # (B,dim,L)
        x = torch.relu(self.conv(x))                 # (B,64,L-2)
        x = self.pool(x)                             # (B,64,L')
        x = self.bn1(x)
        x = self.drop1(x).transpose(1, 2)            # (B,L',64)
        x, _ = self.lstm(x)                          # (B,L',2H)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop2(x)
        e = self.attn_v(torch.tanh(self.attn_W(x))).squeeze(-1)  # (B,L')
        a = torch.softmax(e, dim=1)                              # (B,L')
        ctx = torch.bmm(a.unsqueeze(1), x).squeeze(1)            # (B,2H)
        return torch.sigmoid(self.dense1(ctx))                   # Dense(16, sigmoid)

    def forward(self, ids, attn, feat=None):
        z = self.encode(ids, attn)
        if self.extra_dim and feat is not None:
            z = torch.cat([z, feat], dim=-1)
        return {
            "lethality_logit": self.lethality_head(z),
            "phenotype_logit": self.phenotype_head(z),
            "embedding": z,
        }


def gc_length_features(ids, attn):
    """Trivial features per sequence: [GC fraction, length/640, A,C,G,T fracs].
    Returns (B,6) tensor — for a linear/MLP lower-bound baseline."""
    real = attn.float()
    n = real.sum(1).clamp(min=1)
    def frac(base):
        return ((ids == base).float() * real).sum(1) / n
    a, c, g, t = frac(1), frac(2), frac(3), frac(4)
    gc = c + g
    length = real.sum(1) / ids.size(1)
    return torch.stack([gc, length, a, c, g, t], dim=1)
