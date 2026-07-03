"""Figures 3, 4, 5, 7 and result tables for the binary dsRNA-vs-mRNA task (seaborn).

Manuscript figure numbering (Fig 1 = framework schematic, placeholder):
  fig3 - positive vs negative separability (from manifest_binary.csv)
  fig4 - ROC + PR curves (hero, from val_predictions.csv)
  fig5 - model scale (left) + robustness split/threshold (right), merged
  fig7 - length effect (from val_predictions.csv)
No in-figure titles (captions live in LaTeX). Unified palette, 600 dpi.
Our model is BeetleFormer; baseline is dsRNAPredictor.
"""
import os
import csv
import json
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.figstyle import (set_style, save, panel_label,
                              NAVY, BLUE, SKY, PALE, TAN, BROWN, HERO,
                              BASELINE, POS, NEG, DIVERGE)

DATA = "data/ibeetle_rnai"
CKPT = "checkpoint"
OUT = "paper/figures"
CSVOUT = os.path.join(OUT, "data")
os.makedirs(CSVOUT, exist_ok=True)
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
set_style()


def save_csv(name, header, rowsdata):
    with open(os.path.join(CSVOUT, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rowsdata)


def load_preds(run):
    p = os.path.join(CKPT, run, "val_predictions.csv")
    if not os.path.exists(p):
        return None
    d = {"src": [], "length": [], "label": [], "prob": []}
    for r in csv.DictReader(open(p)):
        d["src"].append(r["src"]); d["length"].append(int(r["length"]))
        d["label"].append(int(r["label"])); d["prob"].append(float(r["prob"]))
    return {"src": d["src"], "length": np.array(d["length"]),
            "label": np.array(d["label"]), "prob": np.array(d["prob"])}


def load_metrics(run):
    p = os.path.join(CKPT, run, "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None


def roc_curve(y, s):
    o = np.argsort(-s); y = y[o]; P = y.sum(); N = len(y) - P
    tpr = np.concatenate([[0], np.cumsum(y)/max(P,1)])
    fpr = np.concatenate([[0], np.cumsum(1-y)/max(N,1)])
    return fpr, tpr, _trapz(tpr, fpr)


def pr_curve(y, s):
    o = np.argsort(-s); y = y[o]; tp = np.cumsum(y); fp = np.cumsum(1-y)
    prec = tp/(tp+fp+1e-12); rec = tp/(y.sum()+1e-12)
    return rec, prec, _trapz(prec, rec)


# ---- Figure 3: positive vs negative separability --------------------------- #
def figure3():
    bpath = os.path.join(DATA, "manifest_binary.csv")
    if not os.path.exists(bpath):
        print("[fig3] no manifest_binary, skip"); return
    pos, neg = [], []
    for r in csv.DictReader(open(bpath)):
        (pos if r["label"]=="1" else neg).append(r["seq"])

    def gc(s): return (s.count("G")+s.count("C"))/max(1,len(s))
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))

    dfL = pd.DataFrame({"length":[len(s) for s in pos]+[len(s) for s in neg],
                        "class":["positive"]*len(pos)+["negative"]*len(neg)})
    sns.histplot(dfL, x="length", hue="class", bins=40, palette={"positive":POS,"negative":NEG},
                 alpha=0.6, edgecolor="white", linewidth=0.2, ax=ax[0,0])
    ax[0,0].set_xlabel("length (bp)"); panel_label(ax[0,0], "a")

    dfG = pd.DataFrame({"gc":[gc(s) for s in pos]+[gc(s) for s in neg],
                        "class":["positive"]*len(pos)+["negative"]*len(neg)})
    sns.kdeplot(dfG, x="gc", hue="class", palette={"positive":POS,"negative":NEG},
                fill=True, alpha=0.4, ax=ax[0,1])
    ax[0,1].set_xlabel("GC fraction"); panel_label(ax[0,1], "b")

    def comp(seqs, b): return np.mean([s.count(b)/max(1,len(s)) for s in seqs])
    dfB = pd.DataFrame([{"base":b,"class":c,"frac":comp(seqs,b)}
                        for b in "ACGT" for c,seqs in [("positive",pos),("negative",neg)]])
    sns.barplot(dfB, x="base", y="frac", hue="class", palette={"positive":POS,"negative":NEG}, ax=ax[0,2])
    ax[0,2].set_ylabel("mean fraction"); panel_label(ax[0,2], "c")
    save_csv("fig3c_basecomp.csv", ["base","class","frac"], dfB.values.tolist())

    dinuc = [a+b for a in "ACGT" for b in "ACGT"]
    def dfreq(seqs):
        c={d:0 for d in dinuc}; tot=0
        for s in seqs:
            for i in range(len(s)-1):
                d=s[i:i+2]
                if d in c: c[d]+=1; tot+=1
        return np.array([c[d]/max(1,tot) for d in dinuc])
    diff=(dfreq(pos)-dfreq(neg)).reshape(4,4)
    sns.heatmap(diff, cmap=DIVERGE, center=0, xticklabels=list("ACGT"), yticklabels=list("ACGT"),
                cbar_kws={"shrink":0.7,"label":"pos - neg"}, ax=ax[1,0])
    panel_label(ax[1,0], "d")

    from itertools import product
    kmers=["".join(p) for p in product("ACGT",repeat=3)]
    def kfreq(seqs):
        c={k:0 for k in kmers}; tot=0
        for s in seqs:
            for i in range(len(s)-2):
                k=s[i:i+3]
                if k in c: c[k]+=1; tot+=1
        return np.array([c[k]/max(1,tot) for k in kmers])
    ratio=np.log2((kfreq(pos)+1e-6)/(kfreq(neg)+1e-6))
    idx=np.argsort(ratio); top=list(idx[-8:][::-1])+list(idx[:8])
    dfK=pd.DataFrame({"kmer":[kmers[i] for i in top],"log2ratio":[ratio[i] for i in top],
                      "sign":["enriched" if ratio[i]>0 else "depleted" for i in top]})
    sns.barplot(dfK, y="kmer", x="log2ratio", hue="sign",
                palette={"enriched":POS,"depleted":NEG}, dodge=False, legend=False, ax=ax[1,1])
    ax[1,1].set_xlabel("log2(pos/neg)"); ax[1,1].tick_params(axis="y", labelsize=6)
    panel_label(ax[1,1], "e")

    lab_f=["positive","negative"]
    sns.barplot(x=lab_f, y=[len(pos),len(neg)], hue=lab_f, palette=[POS,NEG], legend=False, ax=ax[1,2])
    ax[1,2].set_ylabel("count"); ax[1,2].set_xlabel(""); panel_label(ax[1,2], "f")

    for a in ax.flat: sns.despine(ax=a)
    save(fig, os.path.join(OUT, "fig3_separability")); print("[fig3] saved")


# ---- Figure 4: ROC + PR (hero) --------------------------------------------- #
def figure4():
    runs=[("2_1_transformer","BeetleFormer",HERO),
          ("2_3_dsrnapredictor","dsRNAPredictor",BASELINE),
          ("3_2_nofeat","BeetleFormer w/o features",BLUE)]
    have=[(r,l,c) for r,l,c in runs if load_preds(r)]
    if not have: print("[fig4] no preds, skip"); return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    for run,lab,c in have:
        d=load_preds(run)
        fpr,tpr,auc=roc_curve(d["label"],d["prob"]); rec,prec,ap=pr_curve(d["label"],d["prob"])
        ax[0].plot(fpr,tpr,color=c,lw=2.2,label=f"{lab} (AUC={auc:.3f})")
        ax[1].plot(rec,prec,color=c,lw=2.2,label=f"{lab} (AP={ap:.3f})")
        save_csv(f"fig4_roc_{run}.csv", ["fpr","tpr"], list(zip(fpr,tpr)))
    ax[0].plot([0,1],[0,1],ls=":",color="grey",lw=0.8)
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].legend(fontsize=7, loc="lower right"); panel_label(ax[0],"a")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    ax[1].legend(fontsize=7, loc="lower left"); panel_label(ax[1],"b")
    for a in ax: sns.despine(ax=a)
    save(fig, os.path.join(OUT, "fig4_roc_pr")); print("[fig4] saved")


# ---- Figure 5: scale (left) + robustness (right), merged ------------------- #
def figure5():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) scale
    scale=[("3_5a_dim96",96),("2_1_transformer",192),("3_5b_dim384",384)]
    pts=[(dim,load_metrics(r)["final"]["AUC"],load_metrics(r)["final"]["MCC"])
         for r,dim in scale if load_metrics(r)]
    if len(pts)>=2:
        pts.sort(); df=pd.DataFrame(pts,columns=["dim","AUC","MCC"])
        sns.lineplot(df,x="dim",y="AUC",marker="o",color=NAVY,lw=2,markersize=9,label="AUC",ax=ax[0])
        sns.lineplot(df,x="dim",y="MCC",marker="s",color=TAN,lw=2,markersize=9,label="MCC",ax=ax[0])
        ax[0].set_xscale("log",base=2); ax[0].set_xticks(df["dim"]); ax[0].set_xticklabels(df["dim"])
        ax[0].set_xlabel("embedding dimension"); ax[0].set_ylabel("score"); ax[0].legend(fontsize=8)
        save_csv("fig5a_scale.csv",["dim","AUC","MCC"],pts)
    panel_label(ax[0],"a")

    # (b) random vs gene-disjoint ROC
    for run,lab,c in [("2_1_transformer","random split",NAVY),
                      ("4_1_genedisjoint","gene-disjoint split",TAN)]:
        d=load_preds(run)
        if d:
            fpr,tpr,auc=roc_curve(d["label"],d["prob"])
            ax[1].plot(fpr,tpr,color=c,lw=2.2,label=f"{lab} (AUC={auc:.3f})")
    ax[1].plot([0,1],[0,1],ls=":",color="grey",lw=0.8)
    ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate")
    ax[1].legend(fontsize=7,loc="lower right"); panel_label(ax[1],"b")

    # (c) threshold sensitivity
    tpts=[(t,load_metrics(r)["final"]["AUC"],load_metrics(r)["final"]["Acc"])
          for r,t in [("2_1_transformer",20),("4_2a_t30",30),("4_2b_t50",50)] if load_metrics(r)]
    if tpts:
        tpts.sort(); df=pd.DataFrame(tpts,columns=["thr","AUC","Acc"])
        sns.lineplot(df,x="thr",y="AUC",marker="o",color=NAVY,lw=2,markersize=9,label="AUC",ax=ax[2])
        sns.lineplot(df,x="thr",y="Acc",marker="s",color=TAN,lw=2,markersize=9,label="Acc",ax=ax[2])
        ax[2].set_xlabel("positive lethality threshold (%)"); ax[2].set_ylabel("score")
        ax[2].set_xticks([20,30,50]); ax[2].legend(fontsize=8)
        save_csv("fig5c_threshold.csv",["threshold","AUC","Acc"],tpts)
    panel_label(ax[2],"c")

    for a in ax: sns.despine(ax=a)
    save(fig, os.path.join(OUT, "fig5_scale_robustness")); print("[fig5] saved")


# ---- Figure 7: length effect ----------------------------------------------- #
def figure7():
    d=load_preds("2_1_transformer")
    if not d: print("[fig7] main run missing, skip"); return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.set_constrained_layout_pads(w_pad=0.15, wspace=0.12)
    groups=[("130-399",d["length"]<=399),("400-616",d["length"]>399)]
    aucs=[]
    for name,mask in groups:
        if mask.sum()>10 and len(np.unique(d["label"][mask]))==2:
            _,_,auc=roc_curve(d["label"][mask],d["prob"][mask])
            aucs.append((name,auc,int(mask.sum())))
    if aucs:
        df=pd.DataFrame(aucs,columns=["group","AUC","n"])
        sns.barplot(df,x="group",y="AUC",hue="group",palette=[NAVY,BLUE],legend=False,ax=ax[0])
        for i,r in df.iterrows():
            ax[0].text(i,r["AUC"],f"{r['AUC']:.3f}\nn={r['n']}",ha="center",va="bottom",fontsize=7)
        ax[0].set_ylim(0.8,1.0); ax[0].set_ylabel("AUC"); ax[0].set_xlabel("length group (bp)")
        save_csv("fig7a_lengthgroup.csv",["group","AUC","n"],aucs)
    panel_label(ax[0],"a")
    dfS=pd.DataFrame({"length":d["length"],"prob":d["prob"],
                      "class":np.where(d["label"]==1,"positive","negative")})
    sns.scatterplot(dfS,x="length",y="prob",hue="class",palette={"positive":POS,"negative":NEG},
                    s=18,alpha=0.4,edgecolor=None,ax=ax[1])
    ax[1].set_xlabel("length (bp)"); ax[1].set_ylabel("predicted P(effective)")
    ax[1].legend(fontsize=7,title=None); panel_label(ax[1],"b")
    for a in ax: sns.despine(ax=a)
    save(fig, os.path.join(OUT, "fig7_length")); print("[fig7] saved")


# ---- Tables ---------------------------------------------------------------- #
def tables():
    runs=sorted(glob.glob(os.path.join(CKPT,"*","metrics.json")))
    if not runs: print("[tables] none, skip"); return
    header=["run","arch","split","kmer","features","manifest",
            "Sn","Sp","Acc","MCC","Precision","F1","AUC"]
    rows=[]
    for p in runs:
        m=json.load(open(p)); f=m["final"]
        rows.append([m["run_name"],m["arch"],m["split"],m["kmer"],m["use_features"],
                     m["manifest"],round(f["Sn"],4),round(f["Sp"],4),round(f["Acc"],4),
                     round(f["MCC"],4),round(f["Precision"],4),round(f["F1"],4),round(f["AUC"],4)])
    save_csv("all_metrics.csv",header,rows)
    with open(os.path.join(CSVOUT,"all_metrics.md"),"w") as f:
        f.write("| "+" | ".join(header)+" |\n|"+"|".join(["---"]*len(header))+"|\n")
        for r in rows: f.write("| "+" | ".join(str(x) for x in r)+" |\n")
    print(f"[tables] wrote all_metrics ({len(rows)} runs)")


if __name__ == "__main__":
    figure3(); figure4(); figure5(); figure7(); tables()
    print("plot_binary done ->", OUT)
