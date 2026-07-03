"""Evaluation + baselines comparison for RNAi activity prediction.

Loads a trained checkpoint and reports full lethality + phenotype metrics on the
gene-disjoint validation split. Also computes the trivial GC/length ridge
baseline as a lower bound, so the learned encoders can be put in context.

Usage:
    python scripts/evaluate.py --config config/rnai.yaml --ckpt checkpoint/rnai/best.pt
    python scripts/evaluate.py --config config/rnai.yaml --baseline-only
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config, apply_overrides
from utils.device import get_device
from utils.logger import get_logger
from utils import metrics
from data.rnai_dataset import (RNAiDataset, collate, gene_disjoint_split,
                               load_topics, LETH_COLS, _read_manifest,
                               kmer_vocab_size)
from data.features import FEATURE_DIM
from model.seq_transformer import dsRNATransformer
from model.baselines import CNNBiLSTM, gc_length_features


def build_model(cfg, num_topics):
    m = cfg.model
    kmer = int(cfg.data.get("kmer", 1))
    vsz = kmer_vocab_size(kmer)
    extra = FEATURE_DIM if cfg.data.get("use_features", False) else 0
    if m.arch == "cnn_bilstm":
        return CNNBiLSTM(dim=m.get("dim", 128), num_lethality=len(LETH_COLS),
                         num_topics=num_topics, vocab_size=vsz, extra_dim=extra)
    return dsRNATransformer(
        dim=m.dim, depth=m.depth, heads=m.heads, mlp_ratio=m.mlp_ratio,
        dropout=m.dropout, max_len=cfg.data.max_len, num_topics=num_topics,
        num_lethality=len(LETH_COLS), pool=m.get("pool", "mean"),
        conv_stem=m.get("conv_stem", False), stem_kernel=m.get("stem_kernel", 9),
        stem_layers=m.get("stem_layers", 2), stem_pool=m.get("stem_pool", 4),
        vocab_size=vsz, extra_dim=extra,
    )


def collect(loader):
    """Gather features/targets needed for the ridge baseline."""
    feats, lt, lm = [], [], []
    for batch in loader:
        feats.append(gc_length_features(batch["ids"], batch["attn"]).numpy())
        lt.append(batch["lethality"].numpy())
        lm.append(batch["lethality_mask"].numpy())
    return np.concatenate(feats), np.concatenate(lt), np.concatenate(lm)


def ridge_baseline(train_loader, val_loader):
    """Closed-form ridge per lethality column on GC/length features (observed
    entries only). Reports Spearman on val."""
    Xtr, Ytr, Mtr = collect(train_loader)
    Xva, Yva, Mva = collect(val_loader)
    Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    Xva = np.concatenate([Xva, np.ones((len(Xva), 1))], 1)
    rep = {}
    lam = 1.0
    for c, name in enumerate(LETH_COLS):
        mtr = Mtr[:, c]
        if mtr.sum() < 10:
            continue
        A = Xtr[mtr]
        y = Ytr[mtr, c]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
        mva = Mva[:, c]
        pred = Xva[mva] @ w
        rep[f"{name}_spearman"] = metrics.spearman(pred, Yva[mva, c])
    return rep


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    lp, lt, lm, pp, pt = [], [], [], [], []
    for batch in loader:
        feat = batch["feat"].to(device) if "feat" in batch else None
        out = model(batch["ids"].to(device), batch["attn"].to(device), feat)
        lp.append(torch.sigmoid(out["lethality_logit"]).cpu().numpy())
        lt.append(batch["lethality"].numpy())
        lm.append(batch["lethality_mask"].numpy())
        pp.append(out["phenotype_logit"].cpu().numpy())
        pt.append(batch["phenotype"].numpy())
    lp, lt, lm = np.concatenate(lp), np.concatenate(lt), np.concatenate(lm)
    pp, pt = np.concatenate(pp), np.concatenate(pt)
    rep = metrics.lethality_report(lp, lt, lm, LETH_COLS)
    rep.update({f"pheno_{k}": v for k, v in metrics.phenotype_report(pp, pt).items()})
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/rnai.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    device = get_device(cfg.device)
    logger = get_logger(cfg.run_name + "_eval", cfg.paths.out_dir)

    topics = load_topics(cfg.data.data_dir)
    sub = cfg.data.get("pheno_topics", "")
    if sub:
        keep = [t.strip() for t in str(sub).split(",") if t.strip()]
        topics = [t for t in topics if t in keep]
    rows = _read_manifest(cfg.data.manifest)
    lo, hi = int(cfg.data.get("len_min", 0)), int(cfg.data.get("len_max", 10**9))
    if lo > 0 or hi < 10**9:
        rows = [r for r in rows if lo <= len(r["seq"]) <= hi]
    train_idx, val_idx = gene_disjoint_split(rows, cfg.data.val_fraction, cfg.seed)
    kmer = int(cfg.data.get("kmer", 1))
    use_feat = bool(cfg.data.get("use_features", False))
    full = RNAiDataset(cfg.data.manifest, topics, cfg.data.max_len, rows=rows,
                       kmer=kmer, use_features=use_feat)
    train_loader = DataLoader(Subset(full, train_idx), batch_size=cfg.train.batch_size,
                              shuffle=False, collate_fn=collate)
    val_loader = DataLoader(Subset(full, val_idx), batch_size=cfg.train.batch_size,
                            shuffle=False, collate_fn=collate)

    if kmer == 1:
        logger.info("=== GC/length ridge baseline ===")
        base = ridge_baseline(train_loader, val_loader)
        logger.info(json.dumps({k: round(v, 4) for k, v in base.items()}, ensure_ascii=False))
    else:
        logger.info("=== GC/length ridge baseline skipped (kmer>1) ===")

    if args.baseline_only:
        return

    ckpt_path = args.ckpt or os.path.join(cfg.paths.ckpt_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        logger.info(f"no checkpoint at {ckpt_path}; baseline only.")
        return
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg, len(topics)).to(device)
    model.load_state_dict(ck["model"])
    logger.info(f"=== model ({cfg.model.arch}) from {ckpt_path} (step {ck.get('step')}) ===")
    rep = evaluate_model(model, val_loader, device)
    logger.info(json.dumps({k: round(v, 4) for k, v in rep.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
