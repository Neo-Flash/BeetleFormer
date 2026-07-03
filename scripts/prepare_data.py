"""Build a unified RNAi manifest keyed by dsRNA (iB id).

Joins the four iBeetle download files into one table, one row per dsRNA:

    iB_sequence.csv      iB -> dsRNA sequence (+ primers)        [input X]
    iBeetle_lethality.csv iB -> dali11/dali22/dapi11 (%)         [main task y]
    phenotypes.ndjson    gene -> phenotypes, each tagged with    [aux task y]
                         its dsRNA.name (iB) and iBeetleTopic
    iB_TC.csv            iB -> TC gene id                        [grouping only]

Lethality has missing entries (dali22 ~69% missing); we keep them as empty so
the dataset/loss can mask them. Phenotype topics become a fixed 15-class
multi-hot vector (aggregated per dsRNA across all its phenotype annotations).

Output: data/ibeetle_rnai/manifest.csv with columns
    iB, tc, seq, dali11, dali22, dapi11, phase, <one column per topic>

Usage:
    python scripts/prepare_data.py --data-dir data/ibeetle_rnai
"""
import os
import csv
import json
import argparse
from collections import defaultdict


def read_sequences(path):
    """iB -> seq. Header: #iB,seq,leftPrimer,rightPrimer"""
    seqs = {}
    with open(path) as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            if len(row) >= 2 and row[1]:
                seqs[row[0]] = row[1].strip().upper()
    return seqs


def read_lethality(path):
    """iB -> {dali11,dali22,dapi11,phase}. Semicolon-delimited, UTF-8-BOM."""
    leth = {}
    with open(path, encoding="utf-8-sig") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            leth[row["iB"]] = {
                "dali11": row.get("dali11", ""),
                "dali22": row.get("dali22", ""),
                "dapi11": row.get("dapi11", ""),
                "phase": row.get("phase", ""),
            }
    return leth


def read_ib_tc(path):
    """iB -> TC gene id. Header: #iB,TC"""
    m = {}
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                m[parts[0]] = parts[1]
    return m


def read_phenotypes(path):
    """iB -> set of iBeetleTopic. The ndjson is gene-level; each phenotype
    carries its dsRNA.name (iB) so we can attribute topics to the exact dsRNA."""
    ib_topics = defaultdict(set)
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for p in rec.get("phenotypes", []):
                ds = p.get("dsRNA")
                ib = ds.get("name") if isinstance(ds, dict) else ds
                topic = p.get("iBeetleTopic")
                if ib and topic:
                    ib_topics[ib].add(topic)
    return ib_topics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/ibeetle_rnai")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.data_dir
    out_path = args.out or os.path.join(d, "manifest.csv")

    seqs = read_sequences(os.path.join(d, "iB_sequence.csv"))
    leth = read_lethality(os.path.join(d, "iBeetle_lethality.csv"))
    ib_tc = read_ib_tc(os.path.join(d, "iB_TC.csv"))
    ib_topics = read_phenotypes(os.path.join(d, "phenotypes.ndjson"))
    print(f"[prep] sequences={len(seqs)} lethality={len(leth)} "
          f"ib_tc={len(ib_tc)} dsRNA_with_phenotype={len(ib_topics)}")

    # fixed topic vocabulary (sorted for determinism)
    all_topics = sorted({t for ts in ib_topics.values() for t in ts})
    print(f"[prep] phenotype topics ({len(all_topics)}): {all_topics}")

    # a dsRNA is usable if it has BOTH a sequence and at least one lethality value
    leth_cols = ["dali11", "dali22", "dapi11"]
    rows = []
    n_no_seq = n_no_leth = 0
    for ib, seq in seqs.items():
        l = leth.get(ib)
        if l is None or all(l[c] in ("", "NA") for c in leth_cols):
            n_no_leth += 1
            continue
        topics = ib_topics.get(ib, set())
        row = {
            "iB": ib,
            "tc": ib_tc.get(ib, ""),
            "seq": seq,
            "dali11": l["dali11"], "dali22": l["dali22"], "dapi11": l["dapi11"],
            "phase": l["phase"],
        }
        for t in all_topics:
            row[t] = 1 if t in topics else 0
        rows.append(row)

    fields = ["iB", "tc", "seq"] + leth_cols + ["phase"] + all_topics
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # coverage report
    n_with_pheno = sum(1 for r in rows if any(r[t] for t in all_topics))
    n_with_gene = sum(1 for r in rows if r["tc"])
    print(f"[prep] wrote {out_path}: {len(rows)} dsRNA rows "
          f"({n_no_leth} dropped for no lethality)")
    print(f"[prep]   with gene id: {n_with_gene} | with >=1 phenotype: {n_with_pheno}")
    for c in leth_cols:
        n = sum(1 for r in rows if r[c] not in ("", "NA"))
        print(f"[prep]   {c} observed: {n}/{len(rows)} ({100*n//len(rows)}%)")
    # also write the topic vocab for the dataset to load
    with open(os.path.join(d, "topics.txt"), "w") as f:
        f.write("\n".join(all_topics))


if __name__ == "__main__":
    main()
