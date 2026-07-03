"""Generate EXPECTED / illustrative figures so the manuscript is complete before
the real experiments finish. Uses synthetic but plausible data; every output
filename matches the real plotting scripts, so running run_all.sh later OVERWRITES
these mock figures with the real ones.

Expected operating points (honest, gene-disjoint lower than random):
  dsRNATransformer (random split):  Acc~0.86  AUC~0.92  MCC~0.72
  CNN-BiLSTM baseline:              Acc~0.83  AUC~0.89
  logistic-regression lower bound:  Acc~0.75  AUC~0.81
  gene-disjoint split:              AUC~0.85
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREEN, RED, PURPLE = "#3b6fb0", "#e08a1e", "#3a9d5d", "#c0442e", "#7d5ba6"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 130})
rng = np.random.default_rng(508)


def scores_for_auc(n, auc, seed):
    """Synthesize pos/neg scores with an approximate target AUC."""
    r = np.random.default_rng(seed)
    # separation d' chosen from target AUC
    from math import erf, sqrt
    # invert AUC = Phi(d/sqrt2); approximate d
    d = {0.81: 1.25, 0.85: 1.47, 0.89: 1.74, 0.92: 1.99}.get(round(auc, 2), 1.8)
    pos = r.normal(d, 1.0, n); neg = r.normal(0.0, 1.0, n)
    s = np.concatenate([pos, neg]); y = np.concatenate([np.ones(n), np.zeros(n)])
    s = 1 / (1 + np.exp(-(s - d / 2)))  # squash to [0,1]
    return y, s


def roc(y, s):
    o = np.argsort(-s); y = y[o]; P = y.sum(); N = len(y) - P
    tpr = np.concatenate([[0], np.cumsum(y) / P]); fpr = np.concatenate([[0], np.cumsum(1 - y) / N])
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return fpr, tpr, trap(tpr, fpr)


def pr(y, s):
    o = np.argsort(-s); y = y[o]; tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / (tp + fp + 1e-9); rec = tp / (y.sum() + 1e-9)
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return rec, prec, trap(prec, rec)


def note(fig):
    fig.text(0.5, 0.005, "ILLUSTRATIVE / EXPECTED RESULTS — to be replaced with real runs",
             ha="center", fontsize=7, color=RED, style="italic")


# ---- Fig 3: ROC + PR ----
def fig3():
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for auc, lab, c, sd in [(0.92, "dsRNATransformer", BLUE, 1),
                            (0.89, "CNN-BiLSTM", ORANGE, 2),
                            (0.81, "logistic reg. (lower bound)", GREEN, 3)]:
        y, s = scores_for_auc(1500, auc, sd)
        fpr, tpr, a = roc(y, s); rec, prc, ap = pr(y, s)
        ax[0].plot(fpr, tpr, color=c, label=f"{lab} (AUC={a:.3f})")
        ax[1].plot(rec, prc, color=c, label=f"{lab} (AP={ap:.3f})")
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].set_title("(a) ROC"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    ax[1].set_title("(b) Precision-Recall"); ax[1].legend(fontsize=7)
    fig.suptitle("Figure 3. Binary classification performance", fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]); note(fig)
    for e in ("png", "pdf"): fig.savefig(f"{OUT}/fig3_roc_pr.{e}", bbox_inches="tight")
    plt.close(fig)


# ---- Fig 4: scale ----
def fig4():
    dims = [96, 192, 384]; aucs = [0.895, 0.921, 0.918]; mccs = [0.66, 0.72, 0.71]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(dims, aucs, "-o", color=BLUE, label="AUC")
    ax.plot(dims, mccs, "-s", color=ORANGE, label="MCC")
    ax.set_xscale("log", base=2); ax.set_xticks(dims); ax.set_xticklabels(dims)
    ax.set_xlabel("embedding dim"); ax.set_ylabel("score")
    ax.set_title("Figure 4. Effect of model scale"); ax.legend()
    fig.tight_layout(); note(fig)
    for e in ("png", "pdf"): fig.savefig(f"{OUT}/fig4_scale.{e}", bbox_inches="tight")
    plt.close(fig)


# ---- Fig 5: robustness ----
def fig5():
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for auc, lab, c, sd in [(0.92, "random split", BLUE, 5), (0.85, "gene-disjoint split", RED, 6)]:
        y, s = scores_for_auc(1500, auc, sd); fpr, tpr, a = roc(y, s)
        ax[0].plot(fpr, tpr, color=c, label=f"{lab} (AUC={a:.3f})")
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
    ax[0].set_title("(a) Split: random vs gene-disjoint"); ax[0].legend(fontsize=7)
    thr = [20, 30, 50]; auc_t = [0.921, 0.930, 0.945]; acc_t = [0.86, 0.87, 0.89]
    ax[1].plot(thr, auc_t, "-o", color=BLUE, label="AUC")
    ax[1].plot(thr, acc_t, "-s", color=ORANGE, label="Acc")
    ax[1].set_xlabel("dali11 positive threshold (%)"); ax[1].set_ylabel("score")
    ax[1].set_title("(b) Positive-threshold sensitivity"); ax[1].legend(fontsize=7)
    fig.suptitle("Figure 5. Robustness analysis", fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]); note(fig)
    for e in ("png", "pdf"): fig.savefig(f"{OUT}/fig5_robustness.{e}", bbox_inches="tight")
    plt.close(fig)


# ---- Fig 6: interpretability ----
def fig6():
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    W = 60
    diff = rng.normal(0, 0.03, (W, 4)); diff[:8] += rng.normal(0, 0.05, (8, 4))
    im = ax[0, 0].imshow(diff.T, aspect="auto", cmap="coolwarm",
                         vmin=-abs(diff).max(), vmax=abs(diff).max())
    ax[0, 0].set_yticks(range(4)); ax[0, 0].set_yticklabels(list("ACGT"))
    ax[0, 0].set_xlabel("position (nt)"); ax[0, 0].set_title("(a) Base preference (pos - neg)")
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046, pad=0.04)
    sal = np.abs(rng.normal(0, 1, 160)); sal[:20] *= 2.5; sal = np.convolve(sal, np.ones(5)/5, "same")
    ax[0, 1].plot(sal, color=BLUE, lw=0.8)
    ax[0, 1].set_xlabel("token position"); ax[0, 1].set_ylabel("mean |gradient|")
    ax[0, 1].set_title("(b) Input-gradient saliency")
    for lab, c, mu in [(1, GREEN, (2, 0.5)), (0, RED, (-2, -0.5))]:
        pts = rng.normal(mu, 1.2, (500, 2))
        ax[1, 0].scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.4, color=c,
                         label="positive" if lab else "negative")
    ax[1, 0].set_xlabel("PC1 (28.4%)"); ax[1, 0].set_ylabel("PC2 (11.2%)")
    ax[1, 0].set_title("(c) Pooled-embedding PCA"); ax[1, 0].legend(fontsize=7)
    cm = np.array([[1290, 210], [190, 1310]])
    im2 = ax[1, 1].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[1, 1].text(j, i, str(cm[i, j]), ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[1, 1].set_xticks([0, 1]); ax[1, 1].set_xticklabels(["pred neg", "pred pos"])
    ax[1, 1].set_yticks([0, 1]); ax[1, 1].set_yticklabels(["true neg", "true pos"])
    ax[1, 1].set_title("(d) Confusion matrix (thr=0.5)")
    fig.suptitle("Figure 6. Interpretability of the dsRNATransformer", fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97]); note(fig)
    for e in ("png", "pdf"): fig.savefig(f"{OUT}/fig6_interpretability.{e}", bbox_inches="tight")
    plt.close(fig)


# ---- Fig 7: length effect ----
def fig7():
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].bar(["130-399", "400-616"], [0.918, 0.915], color=[BLUE, ORANGE])
    for i, v in enumerate([0.918, 0.915]):
        ax[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax[0].set_ylabel("AUC"); ax[0].set_ylim(0.8, 0.95); ax[0].set_title("(a) AUC by length group")
    for lab, c, mu in [(1, GREEN, 0.72), (0, RED, 0.28)]:
        L = rng.integers(130, 616, 800); p = np.clip(rng.normal(mu, 0.15, 800), 0, 1)
        ax[1].scatter(L, p, s=4, alpha=0.3, color=c, label="positive" if lab else "negative")
    ax[1].set_xlabel("length (bp)"); ax[1].set_ylabel("predicted P(effective)")
    ax[1].set_title("(b) Predicted score vs length"); ax[1].legend(fontsize=7)
    fig.suptitle("Figure 7. Length effect", fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]); note(fig)
    for e in ("png", "pdf"): fig.savefig(f"{OUT}/fig7_length.{e}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig3(); fig4(); fig5(); fig6(); fig7()
    print("expected figures ->", OUT)
