"""Figure 6 - interpretability analysis for the main dsRNATransformer.

Loads checkpoint/2_1_transformer/model.pt and produces four panels:
  (a) position-wise base preference: base frequency by position, positive vs
      negative validation sequences (mirrors OligoFormer base-preference maps);
  (b) input-gradient saliency: mean |d logit / d embedding| per position;
  (c) embedding PCA: 2-D projection of pooled embeddings, coloured by label;
  (d) confusion matrix at threshold 0.5.

Underlying numbers are written to paper/figures/data/*.csv. All labels English.
Skips gracefully if the checkpoint or predictions are absent.
"""
import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.rnai_dataset import encode_seq, kmer_vocab_size
from data.features import compute_features, FEATURE_DIM
from model.seq_transformer import dsRNATransformer

DATA = "data/ibeetle_rnai"
CKPT = "checkpoint/2_1_transformer"
OUT = "paper/figures"
CSVOUT = os.path.join(OUT, "data")
os.makedirs(CSVOUT, exist_ok=True)
from scripts.figstyle import (NAVY, BLUE, SKY, TAN, BROWN, POS, NEG, DIVERGE,
                              SEQ_CMAP, panel_label)
RED, GREEN = TAN, BLUE   # legacy names mapped to unified palette
MAXN = 1500   # cap sequences used for the analysis for speed


def save_csv(name, header, rows):
    with open(os.path.join(CSVOUT, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def main():
    ck = os.path.join(CKPT, "model.pt")
    pred = os.path.join(CKPT, "val_predictions.csv")
    if not (os.path.exists(ck) and os.path.exists(pred)):
        print("[fig6] checkpoint or predictions missing, skip"); return

    meta = json.load(open(os.path.join(CKPT, "metrics.json")))
    cfg = meta["config"]; mcfg = cfg["model"]; dcfg = cfg["data"]
    kmer = int(dcfg.get("kmer", 1)); max_len = int(dcfg.get("max_len", 640))
    use_feat = bool(dcfg.get("use_features", False))

    # map validation ids -> sequence + label (from manifests)
    seqmap = {}
    for r in csv.DictReader(open(os.path.join(DATA, dcfg.get("binary_manifest",
                                                             "manifest_binary.csv")))):
        seqmap[r["id"]] = (r["seq"], int(r["label"]))
    # balanced subsample: the predictions file is ordered positives-then-
    # negatives, so a plain head slice would be almost all positives. Sample
    # each class separately so the PCA / base-preference panels are balanced.
    _allvp = list(csv.DictReader(open(pred)))
    _pos = [r for r in _allvp if int(r["label"]) == 1]
    _neg = [r for r in _allvp if int(r["label"]) == 0]
    half = MAXN // 2
    vp = _pos[:half] + _neg[:half]

    device = torch.device("cpu")
    model = dsRNATransformer(
        dim=mcfg["dim"], depth=mcfg["depth"], heads=mcfg["heads"],
        mlp_ratio=mcfg.get("mlp_ratio", 4.0), dropout=0.0, max_len=max_len,
        num_lethality=1, num_topics=1, pool=mcfg.get("pool", "mean"),
        conv_stem=mcfg.get("conv_stem", False), stem_kernel=mcfg.get("stem_kernel", 9),
        stem_layers=mcfg.get("stem_layers", 2), stem_pool=mcfg.get("stem_pool", 4),
        vocab_size=kmer_vocab_size(kmer),
        extra_dim=FEATURE_DIM if use_feat else 0)
    model.load_state_dict(torch.load(ck, map_location=device)["model"])
    model.eval()

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.set_constrained_layout_pads(w_pad=0.18, h_pad=0.14, wspace=0.16, hspace=0.16)

    # ---- (a) position-wise base preference (first 60 nt) ----
    W = 60
    base_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    posfreq = {1: np.zeros((W, 4)), 0: np.zeros((W, 4))}
    cnt = {1: 0, 0: 0}
    for r in vp:
        seq, lab = seqmap.get(r["id"], (None, None))
        if seq is None: continue
        s = seq.upper().replace("U", "T")[:W]
        for i, c in enumerate(s):
            if c in base_idx:
                posfreq[lab][i, base_idx[c]] += 1
        cnt[lab] += 1
    for lab in (0, 1):
        if cnt[lab]:
            posfreq[lab] /= cnt[lab]
    diff = posfreq[1] - posfreq[0]   # positive - negative
    im = ax[0,0].imshow(diff.T, aspect="auto", cmap=DIVERGE,
                        vmin=-abs(diff).max(), vmax=abs(diff).max())
    ax[0,0].set_yticks(range(4)); ax[0,0].set_yticklabels(list("ACGT"))
    ax[0,0].set_xlabel("position (nt)"); ax[0,0].set_ylabel("base"); panel_label(ax[0,0], "a")
    fig.colorbar(im, ax=ax[0,0], fraction=0.046, pad=0.04, label="pos - neg")
    save_csv("fig6a_basepref.csv", ["position", "A_diff", "C_diff", "G_diff", "T_diff"],
             [[i, *diff[i]] for i in range(W)])

    # ---- (b) input-gradient saliency ----
    sal = np.zeros(max_len if kmer == 1 else max_len // kmer)
    ns = 0
    for r in vp[:400]:
        seq, lab = seqmap.get(r["id"], (None, None))
        if seq is None: continue
        ids, attn = encode_seq(seq, max_len, kmer)
        ids = ids.unsqueeze(0); attn = attn.unsqueeze(0)
        emb = model.embed(ids).detach().clone().requires_grad_(True)
        # forward from embedding: replicate encode() path minimally
        x = emb
        if model.stem is not None:
            x, attn2 = model.stem(x, attn)
        else:
            attn2 = attn
        kp = ~attn2
        x = model.pos(x)
        x = model.encoder(x, src_key_padding_mask=kp)
        x = model.norm(x)
        m = attn2.unsqueeze(-1).float()
        z = (x * m).sum(1) / m.sum(1).clamp(min=1)
        if model.extra_dim:
            z = torch.cat([z, torch.from_numpy(compute_features(seq, max_len)).unsqueeze(0)], -1)
        logit = model.lethality_head(z)[0, 0]
        logit.backward()
        g = emb.grad.abs().sum(-1)[0].numpy()   # per input position
        n = min(len(sal), len(g)); sal[:n] += g[:n]; ns += 1
    if ns: sal /= ns
    ax[0,1].plot(sal, color=NAVY, lw=0.9)
    ax[0,1].set_xlabel("token position"); ax[0,1].set_ylabel("mean |gradient|")
    panel_label(ax[0,1], "b")
    save_csv("fig6b_saliency.csv", ["position", "saliency"],
             [[i, sal[i]] for i in range(len(sal))])

    # ---- (c) embedding PCA ----
    embs, labs = [], []
    with torch.no_grad():
        for r in vp:
            seq, lab = seqmap.get(r["id"], (None, None))
            if seq is None: continue
            ids, attn = encode_seq(seq, max_len, kmer)
            z = model.encode(ids.unsqueeze(0), attn.unsqueeze(0))[0].numpy()
            embs.append(z); labs.append(lab)
    embs = np.array(embs); labs = np.array(labs)
    embs_c = embs - embs.mean(0)
    U, S, Vt = np.linalg.svd(embs_c, full_matrices=False)
    pc = embs_c @ Vt[:2].T
    ax[1,0].scatter(pc[labs==1,0], pc[labs==1,1], s=6, alpha=0.4, color=POS, label="positive")
    ax[1,0].scatter(pc[labs==0,0], pc[labs==0,1], s=6, alpha=0.4, color=NEG, label="negative")
    var = (S**2 / (S**2).sum())[:2]
    ax[1,0].set_xlabel(f"PC1 ({var[0]*100:.1f}%)"); ax[1,0].set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax[1,0].legend(fontsize=7); panel_label(ax[1,0], "c")
    save_csv("fig6c_pca.csv", ["pc1", "pc2", "label"],
             [[pc[i,0], pc[i,1], int(labs[i])] for i in range(len(labs))])

    # ---- (d) confusion matrix ----
    y = np.array([int(r["label"]) for r in vp])
    p = np.array([float(r["prob"]) for r in vp])
    pred = (p >= 0.5).astype(int)
    cm = np.array([[int(((pred==0)&(y==0)).sum()), int(((pred==1)&(y==0)).sum())],
                   [int(((pred==0)&(y==1)).sum()), int(((pred==1)&(y==1)).sum())]])
    im2 = ax[1,1].imshow(cm, cmap=SEQ_CMAP, vmin=0, vmax=cm.max())
    thr_txt = cm.max() * 0.55
    for i in range(2):
        for j in range(2):
            ax[1,1].text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=13,
                         fontweight="bold", color="white" if cm[i,j] > thr_txt else NAVY)
    ax[1,1].set_xticks([0,1]); ax[1,1].set_xticklabels(["pred neg", "pred pos"])
    ax[1,1].set_yticks([0,1]); ax[1,1].set_yticklabels(["true neg", "true pos"])
    fig.colorbar(im2, ax=ax[1,1], fraction=0.046, pad=0.04, label="count")
    panel_label(ax[1,1], "d")
    save_csv("fig6d_confusion.csv", ["", "pred_neg", "pred_pos"],
             [["true_neg", cm[0,0], cm[0,1]], ["true_pos", cm[1,0], cm[1,1]]])

    import seaborn as _sns
    for a in ax.flat: _sns.despine(ax=a)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig6_interpretability.{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig); print("[fig6] saved (600dpi svg/pdf/png)")


if __name__ == "__main__":
    main()
