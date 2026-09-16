"""All tabulated numerical results are computed here from sample-level outputs."""
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import ROOT, RESULTS, CONFIG, report


def main():
    split=pd.read_csv(RESULTS/'source_gene_split.csv').fillna('');train=split.partition.eq('train').to_numpy()
    test=split[split.partition.eq('test')].copy();test['length']=test.seq.str.len();test=test.set_index('id')
    features=np.load(ROOT/'data/inputs.npz')['feat'];y=split.label.to_numpy();it=split.partition.eq('test').to_numpy()
    for name,cols in [('length_only',[5]),('composition_length',list(range(6))),('features_only',list(range(30)))]:
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=3000,random_state=CONFIG['seed']))
        clf.fit(features[train][:,cols],y[train]);p=clf.predict_proba(features[it][:,cols])[:,1]
        pd.DataFrame({'id':test.index,'label':y[it],'score':p}).to_csv(RESULTS/(name+'_test.csv'),index=False)
    rng=np.random.default_rng(CONFIG['seed']);matched=[]
    for length,g in test.groupby('length',sort=True):
        pos=g[g.label.eq(1)];neg=g[g.label.eq(0)];n=min(len(pos),len(neg))
        if n:matched.extend(rng.choice(pos.index,n,replace=False));matched.extend(rng.choice(neg.index,n,replace=False))
    pd.DataFrame({'id':matched}).to_csv(RESULTS/'exact_length_matched_test_ids.csv',index=False)
    names=list(CONFIG['models'])+['length_only','composition_length','features_only'];stats={};joined={}
    for name in names:
        file=RESULTS/(name+'_test.csv')
        if not file.exists():raise RuntimeError('Model training/evaluation incomplete: '+name)
        d=pd.read_csv(file).set_index('id').join(test[['source_gene','length']]);assert set(d.index)==set(test.index)
        assert np.array_equal(d.label,test.loc[d.index,'label']);r=report(d.label.to_numpy(),d.score.to_numpy())
        groups=list(d.groupby('source_gene').indices.values());rng=np.random.default_rng(CONFIG['seed']);boots=[]
        label=d.label.to_numpy();scores=d.score.to_numpy()
        for _ in range(500):
            ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
            boots.append(roc_auc_score(label[ix],scores[ix]))
        r['AUC_gene_bootstrap_95CI']=np.quantile(boots,[.025,.975]).tolist()
        m=d.loc[matched];r['exact_length_matched']=report(m.label.to_numpy(),m.score.to_numpy())
        r['length_groups']={f'{lo}-{hi}':report(g.label.to_numpy(),g.score.to_numpy()) for lo,hi in [(130,399),(400,615)] for g in [d[d.length.between(lo,hi)]]}
        if name in CONFIG['models']:
            h=json.loads((RESULTS/(name+'_history.json')).read_text());best=max(h,key=lambda x:x['AUC'])
            r['selected_epoch']=best['epoch'];r['validation_AUC']=best['AUC']
            r['params']=json.loads((RESULTS/(name+'_training_complete.json')).read_text())['params']
        stats[name]=r;joined[name]=d
    d=joined['beetleformer'].join(joined['cnn_bilstm'][['score']],rsuffix='_cnn')
    groups=list(d.groupby('source_gene').indices.values());rng=np.random.default_rng(CONFIG['seed']);delta=[]
    for _ in range(500):
        ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]);g=d.iloc[ix]
        delta.append(roc_auc_score(g.label,g.score)-roc_auc_score(g.label,g.score_cnn))
    stats['paired_AUC_difference']=dict(estimate=stats['beetleformer']['AUC']-stats['cnn_bilstm']['AUC'],gene_bootstrap_95CI=np.quantile(delta,[.025,.975]).tolist(),replicates=500)
    (RESULTS/'manuscript_statistics.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps({k:dict(AUC=v['AUC'],Acc=v['Acc']) for k,v in stats.items() if 'AUC' in v},indent=2))


if __name__=='__main__':main()
