"""Evaluation metrics for RNAi activity prediction.

Lethality (main task) is a multi-output regression with MISSING values per
column (dali22 is ~69% missing), so every regression metric here takes a mask
and only scores observed entries. Phenotype (auxiliary task) is multi-label.

Pure numpy (no scipy dependency).
"""
import numpy as np


# --------------------------------------------------------------------------- #
# regression (lethality) — all mask-aware, computed per column
# --------------------------------------------------------------------------- #
def _rankdata(a):
    """Average ranks (ties averaged), pure numpy."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(pred, target):
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    if len(pred) < 2 or np.allclose(pred, pred[0]) or np.allclose(target, target[0]):
        return 0.0
    rp, rt = _rankdata(pred), _rankdata(target)
    rp = (rp - rp.mean()) / (rp.std() + 1e-12)
    rt = (rt - rt.mean()) / (rt.std() + 1e-12)
    return float(np.mean(rp * rt))


def pearson(pred, target):
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    if len(pred) < 2:
        return 0.0
    p = (pred - pred.mean()) / (pred.std() + 1e-12)
    t = (target - target.mean()) / (target.std() + 1e-12)
    return float(np.mean(p * t))


def rmse(pred, target):
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred, target):
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    return float(np.mean(np.abs(pred - target)))


def r2(pred, target):
    """Coefficient of determination R^2 = 1 - SS_res / SS_tot."""
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    if len(pred) < 2:
        return 0.0
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def lethality_report(pred, target, mask, col_names):
    """Per-column + macro regression metrics over observed (mask=True) entries.

    Args:
        pred, target: (N, C) arrays in [0, 1] (or any scale).
        mask:         (N, C) bool, True where target is observed.
        col_names:    list of C column names (e.g. ['dali11','dali22','dapi11']).
    Returns dict: per-column metrics + 'macro_*' means across columns.
    """
    pred = np.asarray(pred, float)
    target = np.asarray(target, float)
    mask = np.asarray(mask, bool)
    out = {}
    sp, pe, rm, ma, r2s = [], [], [], [], []
    for c, name in enumerate(col_names):
        m = mask[:, c]
        if m.sum() < 2:
            continue
        p, t = pred[m, c], target[m, c]
        s = spearman(p, t)
        out[f"{name}_spearman"] = s
        out[f"{name}_pearson"] = pearson(p, t)
        out[f"{name}_rmse"] = rmse(p, t)
        out[f"{name}_r2"] = r2(p, t)
        out[f"{name}_n"] = int(m.sum())
        sp.append(s); pe.append(pearson(p, t)); rm.append(rmse(p, t))
        ma.append(mae(p, t)); r2s.append(r2(p, t))
    if sp:
        out["macro_spearman"] = float(np.mean(sp))
        out["macro_pearson"] = float(np.mean(pe))
        out["macro_rmse"] = float(np.mean(rm))
        out["macro_mae"] = float(np.mean(ma))
        out["macro_r2"] = float(np.mean(r2s))
    return out


# --------------------------------------------------------------------------- #
# binary classification (effective dsRNA vs mRNA) — dsRNAPredictor-style metrics
# --------------------------------------------------------------------------- #
def _auc_roc(scores, labels):
    """Area under the ROC curve via rank statistic (Mann-Whitney U)."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    sum_pos = ranks[labels == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_report(probs, labels, threshold=0.5):
    """Sn, Sp, Acc, MCC, Precision, F1, AUC at a fixed threshold.

    probs: (N,) predicted P(effective); labels: (N,) in {0,1}.
    """
    probs = np.asarray(probs, float).ravel()
    labels = np.asarray(labels, int).ravel()
    pred = (probs >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    eps = 1e-12
    sn = tp / (tp + fn + eps)
    sp = tn / (tn + fp + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    f1 = 2 * prec * sn / (prec + sn + eps)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
    mcc = (tp * tn - fp * fn) / mcc_den
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Sn": float(sn), "Sp": float(sp), "Acc": float(acc),
        "MCC": float(mcc), "Precision": float(prec), "F1": float(f1),
        "AUC": _auc_roc(probs, labels),
    }


# --------------------------------------------------------------------------- #
# multi-label classification (phenotype, auxiliary)
# --------------------------------------------------------------------------- #
def _auprc(scores, labels):
    """Average precision (area under precision-recall) for one label column."""
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    n_pos = labels.sum()
    if n_pos == 0:
        return float("nan")
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (n_pos + 1e-12)
    # integrate precision over recall increments
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap)


def phenotype_report(logits, labels, threshold=0.5):
    """Multi-label metrics: micro/macro F1 + macro AUPRC.

    Args:
        logits: (N, L) raw scores; labels: (N, L) {0,1}.
    """
    logits = np.asarray(logits, float)
    labels = np.asarray(labels, float)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(float)

    tp = (preds * labels).sum()
    fp = (preds * (1 - labels)).sum()
    fn = ((1 - preds) * labels).sum()
    micro_f1 = 2 * tp / (2 * tp + fp + fn + 1e-12)

    f1s, aps = [], []
    for l in range(labels.shape[1]):
        tp_l = (preds[:, l] * labels[:, l]).sum()
        fp_l = (preds[:, l] * (1 - labels[:, l])).sum()
        fn_l = ((1 - preds[:, l]) * labels[:, l]).sum()
        f1_l = 2 * tp_l / (2 * tp_l + fp_l + fn_l + 1e-12)
        f1s.append(f1_l)
        ap = _auprc(probs[:, l], labels[:, l])
        if not np.isnan(ap):
            aps.append(ap)
    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "macro_auprc": float(np.mean(aps)) if aps else 0.0,
    }
