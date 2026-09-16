"""Rebuild labels, background fragments, their provenance, and the frozen split.

The positive-length distribution is used for proposal sampling. Rejection of
short transcripts means the accepted negative lengths are NOT exactly matched.
The surrogate-negative label is a benchmark convention, not measured inactivity.
"""
import argparse, csv, hashlib, json, random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from common import ROOT, RESULTS, CONFIG
from data.rnai_dataset import encode_seq
from data.features import compute_features
from data.external_records import load_original_external
from data.sequence_isolation import internal_components, external_matches, assert_external_isolation


def fasta(path):
    name, parts = None, []
    for line in path.read_text().splitlines():
        if line.startswith('>'):
            if name: yield name, ''.join(parts)
            name, parts = line[1:], []
        else: parts.append(line.strip())
    if name: yield name, ''.join(parts)


def gene_of(header):
    for token in header.replace('|', ' ').split():
        if token.startswith('TC'): return token.split('-')[0]
    return header.split()[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--rebuild-cache', action='store_true'); args = ap.parse_args()
    rawdir = ROOT / 'data/raw/ibeetle'
    raw = list(csv.DictReader((rawdir/'manifest.csv').open()))
    topics = (rawdir/'topics.txt').read_text().splitlines()
    positives = [r for r in raw if (r['dali11'] not in ('','NA') and float(r['dali11'])>20) or any(r[t]=='1' for t in topics)]
    targets = {r['tc'] for r in positives if r['tc']}; seqs = {r['seq'].upper() for r in positives}
    lengths = [len(r['seq']) for r in positives]
    transcripts = [(h,gene_of(h),s.upper().replace('U','T')) for h,s in fasta(rawdir/'OGS3_mRNA.fasta')
                   if gene_of(h) not in targets and len(s)>=min(lengths)]
    rng = random.Random(508); negative = []; seen = set(); tries = 0
    while len(negative) < len(positives):
        tries += 1
        if tries > len(positives)*50: raise RuntimeError('Insufficient background fragments')
        length = rng.choice(lengths); header,gene,seq = transcripts[rng.randrange(len(transcripts))]
        if len(seq)<length: continue
        start = rng.randrange(len(seq)-length+1); frag = seq[start:start+length]
        if 'N' in frag or frag in seqs or frag in seen: continue
        seen.add(frag)
        negative.append(dict(id=f'mRNA_{len(negative):06d}', source_gene=gene, transcript=header,
                             start0=start, end0=start+length, seq=frag))
    prov = pd.DataFrame(negative)
    expected = pd.read_csv(ROOT/'provenance/negative_source_provenance.csv')
    pd.testing.assert_frame_equal(prov, expected)
    base = pd.DataFrame([dict(id=r['iB'],seq=r['seq'].upper(),label=1,src='dsRNA',gene=r['tc']) for r in positives]
                        + [dict(id=r['id'],seq=r['seq'],label=0,src='mRNA',gene=f'_negmRNA_{i:06d}') for i,r in enumerate(negative)])
    pd.testing.assert_frame_equal(base, pd.read_csv(rawdir/'manifest_binary.csv').fillna(''))
    source = prov.set_index('id').source_gene
    base['source_gene'] = [source.loc[r.id] if r.label==0 else (r.gene or 'unknown_'+r.id) for r in base.itertuples()]
    trans = str.maketrans('ACGT','TGCA')
    # Freeze external identities before any split or model fit. Outcomes are not
    # consulted in grouping, quarantine or model/checkpoint selection.
    external = load_original_external(ROOT/'data/raw')
    external.to_csv(ROOT/'provenance/original_external_records.csv',index=False)
    identities = external[['id','gene','sequence']].copy()
    lock = ROOT/'provenance/external_identity_lock.csv'
    if lock.exists():pd.testing.assert_frame_equal(identities,pd.read_csv(lock))
    else:identities.to_csv(lock,index=False)
    base['group'],links,ncomparisons = internal_components(base)
    hits = external_matches(base,identities)
    # Keep the internal assignment fixed across the two external protocols.
    groups=base.group.to_numpy();labels=base.label.to_numpy()
    trainval,test=next(GroupShuffleSplit(n_splits=1,test_size=CONFIG['split']['test_group_fraction'],random_state=CONFIG['seed']).split(base,labels,groups))
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=CONFIG['split']['validation_remaining_group_fraction'],random_state=CONFIG['split']['second_seed']).split(base.iloc[trainval],labels[trainval],groups[trainval]))
    base['partition']='';base.loc[trainval[a],'partition']='train';base.loc[trainval[b],'partition']='validation';base.loc[test,'partition']='test'
    base.to_csv(ROOT/'provenance/initial_group_assignment.csv',index=False)
    overlap_ids=set(hits.internal_id)
    for protocol,run in [('sequence',ROOT),('gene',ROOT/'strict_gene')]:
        results=run/'results';data=run/'data'
        results.mkdir(parents=True,exist_ok=True);data.mkdir(parents=True,exist_ok=True)
        (run/'checkpoints').mkdir(exist_ok=True);(run/'logs').mkdir(exist_ok=True)
        remove=base.id.isin(overlap_ids)
        if protocol=='gene':remove=remove|base.source_gene.isin(identities.gene)
        excluded=base[remove].copy();frame=base[~remove].reset_index(drop=True)
        links.to_csv(results/'sequence_component_links.csv',index=False)
        hits.to_csv(results/'external_quarantine_matches.csv',index=False)
        excluded.to_csv(results/'external_quarantine_records.csv',index=False)
        isolation=assert_external_isolation(frame,identities,require_genes=protocol=='gene')
        (results/'external_isolation_checks.json').write_text(json.dumps(isolation,indent=2))
        path=results/'source_gene_split.csv'
        if path.exists():pd.testing.assert_frame_equal(frame,pd.read_csv(path).fillna(''))
        else:frame.to_csv(path,index=False)
        y=frame.label.to_numpy()
        if args.rebuild_cache or not (data/'inputs.npz').exists():
            encoded=[encode_seq(s,640,1) for s in frame.seq]
            features=np.stack([compute_features(s,640) for s in frame.seq])
            np.savez_compressed(data/'inputs.npz',ids=torch.stack([x[0] for x in encoded]).numpy(),attn=torch.stack([x[1] for x in encoded]).numpy(),feat=features,y=y)
            np.save(data/'features_rc.npy',np.stack([compute_features(s.translate(trans)[::-1],640) for s in frame.seq]))
        checks=dict(protocol=protocol,sequences=len(frame),positive=int(y.sum()),negative_sources=int(frame.loc[frame.label.eq(0),'source_gene'].nunique()),eligible_transcripts=len(transcripts),negative_proposals=tries,
            split_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            counts={part:dict(n=len(g),positive=int(g.label.sum()),source_genes=g.source_gene.nunique()) for part,g in frame.groupby('partition')},
            exact_reconstruction=True,initial_sequences=len(base),initial_positive=int(base.label.sum()),
            external_excluded_sequences=len(excluded),external_excluded_positive=int(excluded.label.sum()),external_excluded_source_genes=int(excluded.source_gene.nunique()),
            external_identity_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),internal_cross_gene_sequence_links=len(links),internal_candidate_comparisons=ncomparisons,
            dataset_sha256=hashlib.sha256((data/'inputs.npz').read_bytes()).hexdigest())
        (results/'data_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
