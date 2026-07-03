"""Multi-task training for RNAi activity prediction.

Trains the dsRNA Transformer (or CNN+BiLSTM baseline) to jointly predict
lethality (masked regression, main) and phenotype (multi-label BCE, auxiliary)
on a gene-disjoint split. Reports lethality Spearman (headline: dali11) and
phenotype F1 on the validation set.

Usage:
    python scripts/train.py --config config/rnai.yaml
    python scripts/train.py --set model.arch=cnn_bilstm train.total_steps=4000
"""
import os
import sys
import math
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
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
from model.seq_transformer import (dsRNATransformer, masked_regression_loss,
                                   pairwise_ranking_loss)
from model.baselines import CNNBiLSTM


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


def lr_at(step, base_lr, warmup, total):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(1.0, prog)))


@torch.no_grad()
def evaluate(model, loader, device, topics):
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
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = get_device(cfg.device)
    os.makedirs(cfg.paths.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.paths.out_dir, exist_ok=True)
    logger = get_logger(cfg.run_name, cfg.paths.out_dir)
    logger.info(f"device={device} arch={cfg.model.arch}")

    topics = load_topics(cfg.data.data_dir)
    sub = cfg.data.get("pheno_topics", "")
    if sub:
        keep = [t.strip() for t in str(sub).split(",") if t.strip()]
        topics = [t for t in topics if t in keep]
        logger.info(f"phenotype subset: {len(topics)} topics -> {topics}")
    rows = _read_manifest(cfg.data.manifest)
    lo, hi = int(cfg.data.get("len_min", 0)), int(cfg.data.get("len_max", 10**9))
    if lo > 0 or hi < 10**9:
        rows = [r for r in rows if lo <= len(r["seq"]) <= hi]
        logger.info(f"length group [{lo},{hi}] bp: {len(rows)} dsRNA")
    train_idx, val_idx = gene_disjoint_split(rows, cfg.data.val_fraction, cfg.seed)
    full = RNAiDataset(cfg.data.manifest, topics, cfg.data.max_len, rows=rows,
                       kmer=int(cfg.data.get("kmer", 1)),
                       use_features=bool(cfg.data.get("use_features", False)))
    train_ds, val_ds = Subset(full, train_idx), Subset(full, val_idx)
    logger.info(f"dsRNA: total={len(rows)} train={len(train_ds)} val={len(val_ds)} "
                f"topics={len(topics)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.train.num_workers, collate_fn=collate,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.train.num_workers, collate_fn=collate)

    model = build_model(cfg, len(topics)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model params: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)

    step, best = 0, -1.0
    train_iter = iter(train_loader)
    model.train()
    while step < cfg.train.total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg.train.lr, cfg.train.warmup_steps, cfg.train.total_steps)

        feat = batch["feat"].to(device) if "feat" in batch else None
        out = model(batch["ids"].to(device), batch["attn"].to(device), feat)
        leth_loss = masked_regression_loss(
            out["lethality_logit"], batch["lethality"].to(device),
            batch["lethality_mask"].to(device), kind=cfg.train.leth_loss)
        rank_w = float(cfg.train.get("rank_weight", 0.0))
        if rank_w > 0:
            leth_loss = leth_loss + rank_w * pairwise_ranking_loss(
                out["lethality_logit"], batch["lethality"].to(device),
                batch["lethality_mask"].to(device))
        pheno_loss = F.binary_cross_entropy_with_logits(
            out["phenotype_logit"], batch["phenotype"].to(device))
        loss = leth_loss + cfg.train.pheno_weight * pheno_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % cfg.train.log_every == 0:
            logger.info(f"step {step}/{cfg.train.total_steps} loss={loss.item():.4f} "
                        f"(leth={leth_loss.item():.4f} pheno={pheno_loss.item():.4f}) "
                        f"lr={opt.param_groups[0]['lr']:.2e}")

        if step > 0 and step % cfg.train.eval_every == 0:
            rep = evaluate(model, val_loader, device, topics)
            logger.info(f"[eval {step}] LETHALITY dali11_spearman={rep.get('dali11_spearman',0):.4f} "
                        f"macro_spearman={rep.get('macro_spearman',0):.4f} | "
                        f"PHENOTYPE micro_f1={rep.get('pheno_micro_f1',0):.4f} "
                        f"macro_auprc={rep.get('pheno_macro_auprc',0):.4f}")
            # both tasks are co-equal mains: select by their combined score
            headline = rep.get("dali11_spearman", 0) + rep.get("pheno_macro_auprc", 0)
            if headline > best:
                best = headline
                torch.save({"model": model.state_dict(), "cfg": dict(cfg),
                            "step": step, "report": rep},
                           os.path.join(cfg.paths.ckpt_dir, "best.pt"))
            model.train()
        step += 1

    rep = evaluate(model, val_loader, device, topics)
    logger.info(f"[final] {json.dumps({k: round(v,4) for k,v in rep.items()}, ensure_ascii=False)}")
    torch.save({"model": model.state_dict(), "cfg": dict(cfg), "step": step,
                "report": rep}, os.path.join(cfg.paths.ckpt_dir, "last.pt"))
    logger.info(f"best combined (dali11_spearman + pheno_macro_auprc)={best:.4f}")


if __name__ == "__main__":
    main()
