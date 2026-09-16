"""Original exact/containment/50%-of-20-mers isolation criteria.

Both operational classes and both strand orientations are checked. A shared
source gene alone is not sequence contamination. Gene isolation is a separate
protocol. No model scores or experimental outcomes determine exclusions.
"""
from collections import Counter
import pandas as pd
COMP=str.maketrans('ACGT','TGCA')
K=20
THRESHOLD=.5

def rc(s):return s.translate(COMP)[::-1]
def kmers(s):return {s[i:i+K] for i in range(len(s)-K+1)}

def internal_components(frame):
    """Join source genes and sequence-overlap links before internal splitting.

    For internal pairs, apply the original directional 20-mer criterion in
    both directions. The canonical index only finds candidates; final links
    require an orientation-specific 20-mer intersection.
    """
    genes=frame.source_gene.tolist();seqs=frame.seq.tolist()
    parent={g:g for g in genes}
    def find(g):
        while parent[g]!=g:parent[g]=parent[parent[g]];g=parent[g]
        return g
    index={};sets=[];links=[];checked=0
    for i,s in enumerate(seqs):
        forward=kmers(s);reverse={rc(x) for x in forward};sets.append(forward)
        keys={min(x,rc(x)) for x in forward};counts=Counter()
        for k in keys:
            hits=index.get(k)
            if hits is None:continue
            for j in (hits if isinstance(hits,list) else [hits]):
                if genes[i]!=genes[j]:counts[j]+=1
        for j,count in sorted(counts.items()):
            ga,gb=find(genes[i]),find(genes[j])
            if ga==gb or count<.5*THRESHOLD*min(len(forward),len(sets[j])):continue
            checked+=1
            common=max(len(forward&sets[j]),len(reverse&sets[j]))
            if common/min(len(forward),len(sets[j]))>=THRESHOLD:
                parent[max(ga,gb)]=min(ga,gb)
                links.append(dict(id_a=frame.iloc[i].id,id_b=frame.iloc[j].id,gene_a=genes[i],gene_b=genes[j],shared_20mers=common,coverage_a=common/len(forward),coverage_b=common/len(sets[j])))
        for k in keys:
            hits=index.get(k)
            if hits is None:index[k]=i
            elif isinstance(hits,list):hits.append(i)
            else:index[k]=[hits,i]
        if (i+1)%5000==0:print('Sequence grouping',i+1,'/',len(frame),'cross-gene links',len(links),flush=True)
    return [find(g) for g in genes],pd.DataFrame(links,columns=['id_a','id_b','gene_a','gene_b','shared_20mers','coverage_a','coverage_b']),checked

def external_matches(frame,external):
    """External distinct 20-mers are the denominator, as in the original code."""
    ext=list(external.itertuples());index={};sets=[]
    for i,e in enumerate(ext):
        k=kmers(e.sequence);sets.append(k)
        for orientation,keys in [(0,k),(1,{rc(x) for x in k})]:
            for key in keys:index.setdefault(key,[]).append((i,orientation))
    hits=[]
    for r in frame.itertuples():
        counts=Counter()
        for k in kmers(r.seq):counts.update(index.get(k,[]))
        candidate={i for (i,_),n in counts.items() if n/len(sets[i])>=THRESHOLD}
        for i,e in enumerate(ext):
            s=e.sequence;rev=rc(s)
            exact=s==r.seq or rev==r.seq
            contained=s in r.seq or rev in r.seq or r.seq in s or r.seq in rev
            if exact or contained or i in candidate:
                shared=max(counts.get((i,0),0),counts.get((i,1),0))
                hits.append(dict(internal_id=r.id,source_gene=r.source_gene,external_id=e.id,external_gene=e.gene,
                    reason='exact_or_reverse_complement' if exact else ('full_sequence_containment' if contained else 'external_20mer_fraction_at_least_0.5'),external_20mer_fraction=shared/len(sets[i])))
    return pd.DataFrame(hits,columns=['internal_id','source_gene','external_id','external_gene','reason','external_20mer_fraction'])

def assert_external_isolation(frame,external,require_genes=False):
    hits=external_matches(frame,external)
    assert hits.empty,'External sequence isolation failed: '+str(hits.to_dict('records')[:5])
    genes=sorted(set(frame.source_gene)&set(external.gene))
    if require_genes:assert not genes,genes
    return dict(external_genes=int(external.gene.nunique()),external_sequences=len(external),checked_modeling_sequences=len(frame),sequence_matches=0,genes_shared=genes,require_gene_isolation=require_genes,kmer_length=K,external_kmer_fraction_threshold=THRESHOLD,reverse_complements_checked=True,both_classes_checked=True)
