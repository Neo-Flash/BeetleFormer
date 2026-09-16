"""Verify the frozen protocol, isolation edge cases and retained split identities."""
import hashlib,json,random
import pandas as pd
from common import ROOT,RESULTS
from data.external_records import load_original_external
from data.sequence_isolation import external_matches,internal_components,rc

def main():
    rng=random.Random(73091)
    seq=lambda n: ''.join(rng.choices('ACGT',k=n))
    e=seq(400);unrelated=seq(400)
    external=pd.DataFrame([dict(id='query',gene='same_gene',sequence=e)])
    records=[('exact',e),('reverse',rc(e)),('query_contained',seq(30)+e+seq(30)),('record_contained',e[90:290]),('threshold',e[:240]+seq(160)),('below',e[:120]+seq(280)),('gene_only',unrelated)]
    frame=pd.DataFrame([dict(id=name,source_gene='same_gene',seq=s) for name,s in records])
    found=set(external_matches(frame,external).internal_id)
    assert found=={'exact','reverse','query_contained','record_contained','threshold'},found
    # Two partial overlaps connect three different genes transitively. A gene-only
    # link connects an otherwise unrelated sequence from the first source gene.
    a,b,c=seq(400),seq(400),seq(400)
    rows=pd.DataFrame([dict(id='a',source_gene='g1',seq=a),dict(id='b',source_gene='g2',seq=rc(a[:240]+b[240:])),dict(id='c',source_gene='g3',seq=b[240:]+c[:100]),dict(id='d',source_gene='g1',seq=unrelated)])
    groups,links,_=internal_components(rows)
    assert len(set(groups))==1,(groups,links)
    real=load_original_external(ROOT/'data/raw')
    assert real[['id','gene','sequence']].equals(pd.read_csv(ROOT/'provenance/external_identity_lock.csv'))
    initial=pd.read_csv(ROOT/'provenance/initial_group_assignment.csv').set_index('id')
    outcomes={}
    frames={}
    for protocol,run in [('sequence',ROOT),('gene',ROOT/'strict_gene')]:
        frame=pd.read_csv(run/'results/source_gene_split.csv');checks=json.loads((run/'results/data_checks.json').read_text())
        frames[protocol]=frame
        assert frame.id.is_unique and len(frame)==checks['sequences']
        assert frame.set_index('id').partition.equals(initial.loc[frame.id,'partition'])
        assert frame.set_index('id').seq.equals(initial.loc[frame.id,'seq'])
        for col in ['source_gene','group']:
            sets={part:set(g[col]) for part,g in frame.groupby('partition')}
            assert sum(map(len,sets.values()))==len(set.union(*sets.values()))
        assert external_matches(frame,real).empty
        if protocol=='gene':assert not set(frame.source_gene)&set(real.gene)
        assert checks['split_sha256']==hashlib.sha256((run/'results/source_gene_split.csv').read_bytes()).hexdigest()
        outcomes[protocol]=dict(records=len(frame),external_sequence_matches=0,external_genes_in_training=sorted(set(frame[frame.partition.eq('train')].source_gene)&set(real.gene)),internal_gene_and_component_disjoint=True,initial_assignments_preserved=True)
    assert set(frames['gene'].id)<=set(frames['sequence'].id)
    removed=frames['sequence'][~frames['sequence'].id.isin(frames['gene'].id)]
    assert set(removed.source_gene)<=set(real.gene)
    outcomes['edge_cases']=dict(exact_and_reverse=True,bidirectional_containment=True,threshold_and_below_threshold=True,gene_sharing_alone_allowed=True,transitive_internal_groups=True)
    (RESULTS/'protocol_audit.json').write_text(json.dumps(outcomes,indent=2));print(json.dumps(outcomes,indent=2))

if __name__=='__main__':main()
