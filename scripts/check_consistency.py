"""Checks addressing the data leakage, score and reproducibility concerns."""
import hashlib,json,platform,importlib.metadata
import numpy as np
import pandas as pd
import torch
from common import ROOT,RUN_DATA,RESULTS,CHECKPOINTS,CONFIG,build_model,predict
from model.beetleformer import reverse_complement


def main():
    torch.set_num_threads(3)
    split=pd.read_csv(RESULTS/'source_gene_split.csv');a=np.load(RUN_DATA/'inputs.npz')
    checks=json.loads((RESULTS/'data_checks.json').read_text())
    assert len(split)==checks['sequences'] and split.label.sum()==checks['positive']
    assert hashlib.sha256((RUN_DATA/'inputs.npz').read_bytes()).hexdigest()==checks['dataset_sha256']
    assert np.array_equal(split.label.to_numpy(),a['y'])
    trans=str.maketrans('ACGT','TGCA');partitions={}
    for part,g in split.groupby('partition'):
        partitions[part]=dict(components=set(g.group),genes=set(g.source_gene),seq={min(s,s.translate(trans)[::-1]) for s in g.seq})
    for x,y in [('train','validation'),('train','test'),('validation','test')]:
        assert not partitions[x]['genes']&partitions[y]['genes']
        assert not partitions[x]['components']&partitions[y]['components']
        assert not partitions[x]['seq']&partitions[y]['seq']
    from data.external_records import load_original_external
    from data.sequence_isolation import assert_external_isolation
    ext=load_original_external(ROOT/'data/raw')
    isolation=assert_external_isolation(split,ext,require_genes=checks['protocol']=='gene')
    for name in (list(CONFIG['models']) if checks['protocol']=='sequence' else ['beetleformer','cnn_bilstm']):
        checkpoint=torch.load(CHECKPOINTS/(name+'_best.pt'),map_location='cpu',weights_only=False)
        assert checkpoint['split_sha256']==checks['split_sha256']
        assert checkpoint['evaluation_protocol']==checks['protocol']
    idx=np.flatnonzero(split.partition.eq('test'))[:64]
    ids=torch.from_numpy(a['ids'][idx]);mask=torch.from_numpy(a['attn'][idx])
    ft=torch.from_numpy((a['feat'][idx]+np.load(RUN_DATA/'features_rc.npy')[idx])/2)
    ck=torch.load(CHECKPOINTS/'beetleformer_best.pt',map_location='cpu',weights_only=False)
    m=build_model('beetleformer');m.load_state_dict(ck['model']);m.eval()
    rc=reverse_complement(ids,mask);assert torch.equal(reverse_complement(rc,mask),ids)
    with torch.no_grad():
        logit=m(ids,mask,ft)['lethality_logit'];other=m(rc,mask,ft)['lethality_logit']
        gap=float((logit-other).abs().max());assert gap<1e-6
        modified=ids.clone();modified[~mask]=3
        maskgap=float((m(modified,mask,ft)['lethality_logit']-logit).abs().max());assert maskgap<1e-6
    observed=pd.read_csv(RESULTS/'beetleformer_test.csv').set_index('id').loc[split.iloc[idx].id,'score'].to_numpy()
    current=torch.sigmoid(logit[:,0]).numpy();delta=float(np.abs(current-observed).max());assert delta<2e-5
    loss=torch.nn.functional.binary_cross_entropy_with_logits(m(ids[:4],mask[:4],ft[:4])['lethality_logit'][:,0],torch.tensor([0.,1.,0.,1.]))
    loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    result=dict(evaluation_protocol=checks['protocol'],external_isolation=isolation,source_gene_and_sequence_partitions_disjoint=True,reverse_complement_roundtrip=True,
                strand_symmetry_max_logit_difference=gap,padding_mask_max_logit_difference=maskgap,
                saved_prediction_max_absolute_difference=delta,finite_gradients=True,
                model_parameters=sum(p.numel() for p in m.parameters()),test_sequences_checked=len(idx),
                split_sha256=hashlib.sha256((RESULTS/'source_gene_split.csv').read_bytes()).hexdigest())
    (RESULTS/'consistency_checks.json').write_text(json.dumps(result,indent=2))
    packages=['torch','numpy','pandas','scipy','scikit-learn','matplotlib','Pillow','python-docx','lxml','openpyxl','PyYAML','PyMuPDF']
    versions={p:importlib.metadata.version(p) for p in packages}
    (ROOT/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versions.items())+'\n')
    (RESULTS/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),packages=versions,mps_available=torch.backends.mps.is_available()),indent=2))
    (RUN_DATA/'example.fasta').write_text('\n'.join('>'+r.id+'\n'+r.seq for r in split.iloc[idx[:3]].itertuples())+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
