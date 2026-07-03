"""Figure 2 - dataset overview for the binary dsRNA-vs-mRNA task (seaborn).

Six panels, no in-figure titles (captions live in LaTeX): length, dali11
distribution, threshold->positive fraction, dsRNA per gene, positive source,
class balance. 600 dpi, unified palette, constrained layout. Labels English.
"""
import os
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.figstyle import (set_style, save, panel_label,
                              NAVY, BLUE, SKY, TAN, BROWN, POS, NEG)

DATA = "data/ibeetle_rnai"
OUT = "paper/figures"
CSVOUT = os.path.join(OUT, "data")
os.makedirs(CSVOUT, exist_ok=True)
set_style()


def save_csv(name, header, rows):
    with open(os.path.join(CSVOUT, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


rows = list(csv.DictReader(open(os.path.join(DATA, "manifest.csv"))))
topics = [t for t in open(os.path.join(DATA, "topics.txt")).read().splitlines() if t]
lengths = np.array([len(r["seq"]) for r in rows])
d11 = np.array([float(r["dali11"]) if r["dali11"] not in ("", "NA") else np.nan for r in rows])
from collections import Counter
per_gene = np.array(list(Counter(r["tc"] for r in rows if r["tc"]).values()))

fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))

# (a) length
sns.histplot(lengths, bins=45, color=BLUE, edgecolor="white", linewidth=0.3, ax=ax[0,0])
ax[0,0].axvline(399, color=BROWN, ls="--", lw=1.2, label="130-399 / 400-616")
ax[0,0].axvline(int(np.median(lengths)), color=NAVY, ls=":", lw=1.2,
                label=f"median {int(np.median(lengths))} bp")
ax[0,0].set_xlabel("dsRNA length (bp)"); ax[0,0].set_ylabel("count")
ax[0,0].legend(fontsize=6); panel_label(ax[0,0], "a")

# (b) dali11
v = d11[~np.isnan(d11)]
sns.histplot(v, bins=np.arange(0,105,5), color=BLUE, edgecolor="white", linewidth=0.3, ax=ax[0,1])
ax[0,1].axvline(20, color=BROWN, ls="--", lw=1.2, label="20% threshold")
ax[0,1].set_xlabel("dali11 lethality (%)"); ax[0,1].set_ylabel("count")
ax[0,1].legend(fontsize=6); panel_label(ax[0,1], "b")

# (c) threshold -> positive fraction
thr = np.arange(0, 101, 5)
posrate = [100*np.mean(v>t) for t in thr]
sns.lineplot(x=thr, y=posrate, marker="o", color=NAVY, ax=ax[0,2], markersize=5)
for t in (20,30,50):
    ax[0,2].axvline(t, color=TAN, ls="--", lw=0.7)
    ax[0,2].annotate(f"{t}%: {100*np.mean(v>t):.0f}%", (t, 100*np.mean(v>t)),
                     fontsize=6, xytext=(3,4), textcoords="offset points")
ax[0,2].set_xlabel("dali11 threshold (%)"); ax[0,2].set_ylabel("positive fraction (%)")
panel_label(ax[0,2], "c")
save_csv("fig2c_threshold.csv", ["threshold","positive_fraction_pct"], [[t,100*np.mean(v>t)] for t in thr])

# (d) dsRNA per gene
sns.histplot(per_gene, bins=np.arange(0.5, per_gene.max()+1.5, 1), color=BLUE,
             edgecolor="white", linewidth=0.3, ax=ax[1,0])
ax[1,0].set_yscale("log"); ax[1,0].set_xlabel("dsRNA per gene"); ax[1,0].set_ylabel("gene count (log)")
panel_label(ax[1,0], "d")

# (e) positive source
lethal = np.where(np.isnan(d11), False, d11 > 20)
has_ph = np.array([any(r.get(t)=="1" for t in topics) for r in rows])
vals = [int((lethal & ~has_ph).sum()), int((~lethal & has_ph).sum()), int((lethal & has_ph).sum())]
cats = ["lethal\nonly", "phenotype\nonly", "both"]
sns.barplot(x=cats, y=vals, hue=cats, palette=[NAVY, TAN, BLUE], legend=False, ax=ax[1,1])
for i,val in enumerate(vals): ax[1,1].text(i, val, str(val), ha="center", va="bottom", fontsize=7)
ax[1,1].set_ylabel("count"); ax[1,1].set_xlabel(""); panel_label(ax[1,1], "e")
save_csv("fig2e_possource.csv", ["category","count"], list(zip(["lethal_only","pheno_only","both"], vals)))

# (f) class balance
bpath = os.path.join(DATA, "manifest_binary.csv")
npos = nneg = 0
if os.path.exists(bpath):
    for r in csv.DictReader(open(bpath)):
        if r["label"]=="1": npos+=1
        else: nneg+=1
lab_f = ["effective\ndsRNA (pos)", "mRNA\nfragment (neg)"]
sns.barplot(x=lab_f, y=[npos,nneg], hue=lab_f, palette=[POS, NEG], legend=False, ax=ax[1,2])
for i,val in enumerate([npos,nneg]): ax[1,2].text(i, val, str(val), ha="center", va="bottom", fontsize=7)
ax[1,2].set_ylabel("count"); ax[1,2].set_xlabel(""); panel_label(ax[1,2], "f")

for a in ax.flat: sns.despine(ax=a)
save(fig, os.path.join(OUT, "fig2_dataset_overview"))
print("saved fig2 (600dpi svg/pdf/png)")
