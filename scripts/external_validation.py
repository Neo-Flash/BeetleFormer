"""Evaluate the unchanged original 20-positive/20-background external panel.

No model selection, threshold fitting or architecture change uses this panel.
All original records are retained; overlapping modeling records are removed
before training. The former 28-fragment ranking panel is not evaluated here.
"""
import hashlib,json
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score
from common import ROOT,RESULTS,CHECKPOINTS,CONFIG,build_model,predict,report
from data.rnai_dataset import encode_seq
from data.features import compute_features
from data.external_records import load_original_external
from data.sequence_isolation import assert_external_isolation


def cluster_auc_interval(df,logits):
    # Resample whole source genes within each operational class. The same
    # fixed bootstrap draws are used for every model and both protocols.
    groups=[[np.flatnonzero((df.label.to_numpy()==label)&(df.gene.to_numpy()==gene))
             for gene in sorted(df.loc[df.label.eq(label),'gene'].unique())]
            for label in [0,1]]
    rng=np.random.default_rng(CONFIG['seed']); values=[]
    y=df.label.to_numpy()
    for _ in range(5000):
        ix=np.concatenate([g[j] for g in groups for j in rng.integers(0,len(g),len(g))])
        values.append(roc_auc_score(y[ix],logits[ix]))
    return np.quantile(values,[.025,.975]).tolist()


def main():
    torch.set_num_threads(3)
    df=load_original_external(ROOT/'data/raw')
    assert len(df)==40 and df.length.between(130,615).all()
    split=pd.read_csv(RESULTS/'source_gene_split.csv')
    checks=json.loads((RESULTS/'data_checks.json').read_text())
    isolation=assert_external_isolation(split,df,require_genes=checks['protocol']=='gene')
    assert df[['id','gene','sequence']].equals(pd.read_csv(ROOT/'provenance/external_identity_lock.csv'))
    assert hashlib.sha256((RESULTS/'source_gene_split.csv').read_bytes()).hexdigest()==checks['split_sha256']
    seqs=df.sequence.tolist();trans=str.maketrans('ACGT','TGCA')
    encoded=[encode_seq(s,640,1) for s in seqs]
    ids=torch.stack([e[0] for e in encoded]);mask=torch.stack([e[1] for e in encoded])
    ft=np.stack([compute_features(s,640) for s in seqs])
    rc=np.stack([compute_features(s.translate(trans)[::-1],640) for s in seqs])
    models=['beetleformer','cnn_bilstm']+(['single_scale_transformer'] if checks['protocol']=='sequence' else [])
    statistics={}
    for name in models:
        ck=torch.load(CHECKPOINTS/(name+'_best.pt'),map_location='cpu',weights_only=False)
        assert ck['split_sha256']==checks['split_sha256']
        model=build_model(name);model.load_state_dict(ck['model'])
        features=torch.from_numpy((ft+rc)/2 if CONFIG['models'][name]['symmetric_features'] else ft)
        logits=predict(model,ids,mask,features,np.arange(len(df)),'cpu',logits=True)
        scores=expit(logits);y=df.label.to_numpy()
        df[name+'_logit']=logits;df[name+'_score']=scores;df[name+'_predicted_label']=(logits>=0).astype(int)
        r=report(y,scores)
        r.update(AUC=float(roc_auc_score(y,logits)),AP=float(average_precision_score(y,logits)),
            AUC_gene_bootstrap_95CI=cluster_auc_interval(df,logits),
            threshold=.5,selected_epoch=ck['epoch'],
            checkpoint_sha256=hashlib.sha256((CHECKPOINTS/(name+'_best.pt')).read_bytes()).hexdigest())
        statistics[name]=r
    df['source_gene_in_training']=df.gene.isin(split.loc[split.partition.eq('train'),'source_gene'])
    df.to_csv(RESULTS/'external_classification.csv',index=False)
    statistics.update(evaluation_protocol=checks['protocol'],external_n=len(df),positive=int(df.label.sum()),
        negative=int((df.label==0).sum()),external_source_genes=int(df.gene.nunique()),
        external_positive_genes=int(df.loc[df.label.eq(1),'gene'].nunique()),
        source_counts=df.source_paper.value_counts().to_dict(),
        source_genes_shared_with_training=sorted(df.loc[df.source_gene_in_training,'gene'].unique().tolist()),
        isolation=isolation,split_sha256=checks['split_sha256'],
        external_identity_sha256=checks['external_identity_sha256'],
        original_source_sha256=json.loads((ROOT/'data/raw/original_external/source_manifest.json').read_text())['sha256'],
        interpretation='Binary candidate-prescreening evaluation using the unchanged original 20 dsRNA-positive and 20 mRNA-background sequences from four studies and OGS3; both isolation protocols use identical external labels and sequences; higher AUC indicates stronger positive/background discrimination.')
    (RESULTS/'external_statistics.json').write_text(json.dumps(statistics,indent=2))
    print(json.dumps({k:statistics[k] for k in models},indent=2),flush=True)

if __name__=='__main__':main()
