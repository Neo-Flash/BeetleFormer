"""Build a binary dsRNA-vs-mRNA manifest, mirroring the dsRNAPredictor recipe.

Positives : effective dsRNA reagents from the iBeetle manifest, i.e. dali11 > 20%
            OR carrying at least one phenotype annotation. Uses the dsRNA
            fragment sequence (the model input).
Negatives : fragments sampled from Tribolium castaneum OGS3 mRNA transcripts,
            with a length distribution matched to the positives, at a 1:1 ratio.
            Sampled only from transcripts whose gene is NOT a positive target
            (avoids labelling a real dsRNA region as negative), and de-duplicated
            against positive sequences.

Reference (methodology only, implemented independently):
  Cheng et al., dsRNAPredictor; Xu et al., dsRNAPredictor-II (Methods 232, 2024):
  negatives = mRNA sequences, stratified-sampled to the positive length
  distribution.

Output: data/ibeetle_rnai/manifest_binary.csv with columns
        id, seq, label, src   (label 1 = effective dsRNA, 0 = mRNA fragment)

Usage:
    python scripts/prepare_binary.py --data-dir data/ibeetle_rnai --seed 508
"""
import os
import csv
import gzip
import random
import argparse
from collections import Counter


def read_manifest(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_topics(data_dir):
    p = os.path.join(data_dir, "topics.txt")
    return [t for t in open(p).read().splitlines() if t]


def read_fasta(path):
    """Yield (id, seq) from a (optionally gzipped) FASTA. id is the TC gene tag."""
    op = gzip.open if path.endswith(".gz") else open
    name, buf = None, []
    with op(path, "rt") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if name:
                    yield name, "".join(buf)
                name, buf = line[1:], []
            else:
                buf.append(line)
    if name:
        yield name, "".join(buf)


def gene_of(header):
    """Extract the TCxxxxxx gene id from an OGS3 mRNA header like
    'gnl|WGS:AAJJ|TC002748-RA' -> 'TC002748'."""
    for tok in header.replace("|", " ").split():
        if tok.startswith("TC") and "-" in tok:
            return tok.split("-")[0]
        if tok.startswith("TC"):
            return tok
    return header.split()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/ibeetle_rnai")
    ap.add_argument("--seed", type=int, default=508)
    ap.add_argument("--leth-thresh", type=float, default=20.0)
    ap.add_argument("--mrna", default="OGS3_mRNA.fasta")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.data_dir
    rng = random.Random(args.seed)

    rows = read_manifest(os.path.join(d, "manifest.csv"))
    topics = load_topics(d)

    # --- positives: effective dsRNA ---
    pos = []
    pos_genes = set()
    pos_seqs = set()
    for r in rows:
        v = r.get("dali11", "")
        lethal = v not in ("", "NA") and float(v) > args.leth_thresh
        has_pheno = any(r.get(t) == "1" for t in topics)
        if lethal or has_pheno:
            pos.append(r)
            if r.get("tc"):
                pos_genes.add(r["tc"])
            pos_seqs.add(r["seq"].upper())
    pos_lengths = [len(r["seq"]) for r in pos]
    len_hist = Counter(pos_lengths)
    print(f"[bin] positives (effective dsRNA): {len(pos)}  "
          f"len[min/med/max]={min(pos_lengths)}/{sorted(pos_lengths)[len(pos_lengths)//2]}/{max(pos_lengths)}")

    # --- candidate mRNA transcripts from non-positive genes ---
    mrna_path = os.path.join(d, args.mrna)
    transcripts = []
    for hdr, seq in read_fasta(mrna_path):
        g = gene_of(hdr)
        if g in pos_genes:
            continue  # avoid sampling from a real dsRNA target gene
        seq = seq.upper().replace("U", "T")
        if len(seq) >= min(pos_lengths):
            transcripts.append(seq)
    print(f"[bin] usable mRNA transcripts (non-target genes): {len(transcripts)}")

    # --- sample negatives: one fragment per drawn transcript, length matched ---
    # draw a target length from the positive length histogram, then cut a random
    # window of that length from a random transcript long enough to contain it.
    lengths_pool = list(pos_lengths)
    neg = []
    neg_seqs = set()
    tries = 0
    max_tries = len(pos) * 50
    while len(neg) < len(pos) and tries < max_tries:
        tries += 1
        L = rng.choice(lengths_pool)
        t = transcripts[rng.randrange(len(transcripts))]
        if len(t) < L:
            continue
        start = rng.randrange(0, len(t) - L + 1)
        frag = t[start:start + L]
        if "N" in frag:
            continue
        if frag in pos_seqs or frag in neg_seqs:
            continue
        neg_seqs.add(frag)
        neg.append(frag)
    print(f"[bin] negatives sampled: {len(neg)} (tries={tries})")

    # --- write manifest ---
    out_path = args.out or os.path.join(d, "manifest_binary.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "seq", "label", "src", "gene"])
        for r in pos:
            w.writerow([r["iB"], r["seq"].upper(), 1, "dsRNA", r.get("tc", "")])
        for i, s in enumerate(neg):
            # negatives get a synthetic singleton gene so gene-disjoint splitting
            # never places a negative fragment across both partitions.
            w.writerow([f"mRNA_{i:06d}", s, 0, "mRNA", f"_negmRNA_{i:06d}"])
    n = len(pos) + len(neg)
    print(f"[bin] wrote {out_path}: {n} rows ({len(pos)} pos / {len(neg)} neg)")


if __name__ == "__main__":
    main()
