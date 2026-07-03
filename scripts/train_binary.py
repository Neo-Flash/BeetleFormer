"""Binary classification: effective dsRNA vs mRNA fragment (dsRNAPredictor-style).

Reuses the dsRNATransformer / CNNBiLSTM encoders with a single binary head, on
the balanced manifest_binary.csv (random stratified 80/20 split). Reports
Sn/Sp/Acc/MCC/Precision/F1/AUC, matching the dsRNAPredictor evaluation.

Also supports the improvement knobs so we can compare against a plain baseline:
  data.kmer=K            k-mer tokenization
  data.use_features=true concat thermodynamic/tiling features before the head
  model.conv_stem=...    conv front end
  model.arch=cnn_bilstm  the CNN-BiLSTM baseline encoder

Usage:
  python scripts/train_binary.py --set model.arch=transformer model.conv_stem=true
"""
import os
import sys
import csv
import math
import json
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config, apply_overrides
from utils.device import get_device
from utils.logger import get_logger
from utils import metrics
from data.rnai_dataset import encode_seq, kmer_vocab_size, VOCAB_SIZE
from data.features import compute_features, FEATURE_DIM
from model.seq_transformer import dsRNATransformer
from model.baselines import CNNBiLSTM, CNNBiLSTMFaithful


class BinaryDataset(Dataset):
    def __init__(self, rows, max_len, kmer=1, use_features=False):
        self.rows = rows
        self.max_len = max_len
        self.kmer = kmer
        self.use_features = use_features

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ids, attn = encode_seq(r["seq"], self.max_len, self.kmer)
        s = {"ids": ids, "attn": attn, "label": torch.tensor(float(r["label"]))}
        if self.use_features:
            s["feat"] = torch.from_numpy(compute_features(r["seq"], self.max_len))
        return s


def collate(batch):
    out = {
        "ids": torch.stack([b["ids"] for b in batch]),
        "attn": torch.stack([b["attn"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }
    if "feat" in batch[0]:
        out["feat"] = torch.stack([b["feat"] for b in batch])
    return out


def stratified_split(rows, val_fraction, seed):
    """Random stratified 80/20 by label (dsRNAPredictor-style, not gene-disjoint)."""
    by = {0: [], 1: []}
    for i, r in enumerate(rows):
        by[int(r["label"])].append(i)
    rng = random.Random(seed)
    tr, va = [], []
    for lab, idxs in by.items():
        rng.shuffle(idxs)
        k = int(len(idxs) * val_fraction)
        va += idxs[:k]; tr += idxs[k:]
    return sorted(tr), sorted(va)


def gene_disjoint_split(rows, val_fraction, seed):
    """Split so no gene appears in both partitions (rows carry a 'gene' column;
    negatives already have unique singleton gene ids)."""
    by_gene = {}
    for i, r in enumerate(rows):
        key = r.get("gene") or f"_solo_{r['id']}"
        by_gene.setdefault(key, []).append(i)
    genes = sorted(by_gene)
    random.Random(seed).shuffle(genes)
    n_target = int(len(rows) * val_fraction)
    va, tr, nv = [], [], 0
    for g in genes:
        if nv < n_target:
            va += by_gene[g]; nv += len(by_gene[g])
        else:
            tr += by_gene[g]
    return sorted(tr), sorted(va)


def build_model(cfg):
    m = cfg.model
    kmer = int(cfg.data.get("kmer", 1))
    vsz = kmer_vocab_size(kmer)
    extra = FEATURE_DIM if cfg.data.get("use_features", False) else 0
    # single binary head: reuse encoders with num_lethality=1, num_topics=1;
    # we use the lethality_logit as the binary logit and ignore the pheno head.
    if m.arch == "cnn_bilstm":
        return CNNBiLSTM(dim=m.get("dim", 128), num_lethality=1, num_topics=1,
                         vocab_size=vsz, extra_dim=extra)
    if m.arch == "cnn_faithful":
        return CNNBiLSTMFaithful(dim=m.get("dim", 100), num_lethality=1,
                                 num_topics=1, vocab_size=vsz, extra_dim=extra)
    return dsRNATransformer(
        dim=m.dim, depth=m.depth, heads=m.heads, mlp_ratio=m.mlp_ratio,
        dropout=m.dropout, max_len=cfg.data.max_len, num_lethality=1, num_topics=1,
        pool=m.get("pool", "mean"), conv_stem=m.get("conv_stem", False),
        stem_kernel=m.get("stem_kernel", 9), stem_layers=m.get("stem_layers", 2),
        stem_pool=m.get("stem_pool", 4), vocab_size=vsz, extra_dim=extra)


def lr_at(step, base, warm, total):
    if step < warm:
        return base * step / max(1, warm)
    prog = (step - warm) / max(1, total - warm)
    return 0.5 * base * (1 + math.cos(math.pi * min(1.0, prog)))


@torch.no_grad()
def evaluate(model, loader, device, return_preds=False):
    model.eval()
    ps, ys = [], []
    for b in loader:
        feat = b["feat"].to(device) if "feat" in b else None
        out = model(b["ids"].to(device), b["attn"].to(device), feat)
        ps.append(torch.sigmoid(out["lethality_logit"][:, 0]).cpu().numpy())
        ys.append(b["label"].numpy())
    probs = np.concatenate(ps); labels = np.concatenate(ys)
    rep = metrics.binary_report(probs, labels)
    if return_preds:
        return rep, probs, labels
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/rnai.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = get_device(cfg.device)
    os.makedirs(cfg.paths.out_dir, exist_ok=True)
    logger = get_logger(cfg.run_name, cfg.paths.out_dir)

    path = os.path.join(cfg.data.data_dir,
                        cfg.data.get("binary_manifest", "manifest_binary.csv"))
    rows = list(csv.DictReader(open(path)))
    kmer = int(cfg.data.get("kmer", 1))
    use_feat = bool(cfg.data.get("use_features", False))
    split = cfg.data.get("split", "random")
    if split == "gene":
        tr_idx, va_idx = gene_disjoint_split(rows, cfg.data.val_fraction, cfg.seed)
    else:
        tr_idx, va_idx = stratified_split(rows, cfg.data.val_fraction, cfg.seed)
    full = BinaryDataset(rows, cfg.data.max_len, kmer, use_feat)
    tr = DataLoader(Subset(full, tr_idx), batch_size=cfg.train.batch_size, shuffle=True,
                    num_workers=cfg.train.num_workers, collate_fn=collate, drop_last=True)
    va = DataLoader(Subset(full, va_idx), batch_size=cfg.train.batch_size, shuffle=False,
                    num_workers=cfg.train.num_workers, collate_fn=collate)
    logger.info(f"binary: total={len(rows)} train={len(tr_idx)} val={len(va_idx)} "
                f"arch={cfg.model.arch} kmer={kmer} feat={use_feat} split={split} "
                f"manifest={os.path.basename(path)}")

    model = build_model(cfg).to(device)
    logger.info(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    if cfg.train.get("optimizer", "adamw") == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)  # dsRNAPredictor: plain Adam
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                weight_decay=cfg.train.weight_decay)

    step, best = 0, -1.0
    it = iter(tr); model.train()
    while step < cfg.train.total_steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(tr); b = next(it)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg.train.lr, cfg.train.warmup_steps, cfg.train.total_steps)
        feat = b["feat"].to(device) if "feat" in b else None
        out = model(b["ids"].to(device), b["attn"].to(device), feat)
        loss = F.binary_cross_entropy_with_logits(
            out["lethality_logit"][:, 0], b["label"].to(device))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % cfg.train.log_every == 0:
            logger.info(f"step {step}/{cfg.train.total_steps} loss={loss.item():.4f} "
                        f"lr={opt.param_groups[0]['lr']:.2e}")
        if step > 0 and step % cfg.train.eval_every == 0:
            rep = evaluate(model, va, device)
            logger.info(f"[eval {step}] Acc={rep['Acc']:.4f} AUC={rep['AUC']:.4f} "
                        f"MCC={rep['MCC']:.4f} F1={rep['F1']:.4f}")
            if rep["AUC"] > best:
                best = rep["AUC"]
            model.train()
        step += 1

    rep, probs, labels = evaluate(model, va, device, return_preds=True)
    logger.info(f"[final] {json.dumps({k: (round(v,4) if isinstance(v,float) else v) for k,v in rep.items()}, ensure_ascii=False)}")

    # ---- persist raw results for downstream plotting / tables ----
    # every experiment saves: per-validation-sample predictions + ids/labels,
    # the full metric dict, the resolved config, and the model checkpoint.
    os.makedirs(cfg.paths.ckpt_dir, exist_ok=True)
    va_ids = [rows[i]["id"] for i in va_idx]
    va_src = [rows[i].get("src", "") for i in va_idx]
    va_len = [len(rows[i]["seq"]) for i in va_idx]
    pred_path = os.path.join(cfg.paths.ckpt_dir, "val_predictions.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "src", "length", "label", "prob"])
        for _id, _src, _len, y, p in zip(va_ids, va_src, va_len, labels, probs):
            w.writerow([_id, _src, _len, int(y), float(p)])
    with open(os.path.join(cfg.paths.ckpt_dir, "metrics.json"), "w") as f:
        json.dump({"run_name": cfg.run_name, "split": split, "kmer": kmer,
                   "use_features": use_feat, "arch": cfg.model.arch,
                   "manifest": os.path.basename(path), "best_auc": best,
                   "final": rep, "config": dict(cfg)}, f, indent=2, default=str)
    torch.save({"model": model.state_dict(), "cfg": dict(cfg), "report": rep},
               os.path.join(cfg.paths.ckpt_dir, "model.pt"))
    logger.info(f"[saved] preds -> {pred_path} ; metrics.json ; model.pt")


if __name__ == "__main__":
    main()
