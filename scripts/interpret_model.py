"""Regenerate descriptive interpretation data from the frozen BeetleFormer."""
import json
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from common import ROOT, RESULTS, CHECKPOINTS, CONFIG, build_model


def main():
    torch.set_num_threads(3)
    df=pd.read_csv(RESULTS/'source_gene_split.csv');a=np.load(ROOT/'data/inputs.npz')
    ids=torch.from_numpy(a['ids']);mask=torch.from_numpy(a['attn'])
    feat=torch.from_numpy((a['feat']+np.load(ROOT/'data/features_rc.npy'))/2)
    ck=torch.load(CHECKPOINTS/'beetleformer_best.pt',map_location='cpu',weights_only=False)
    assert ck['split_sha256']==json.loads((RESULTS/'data_checks.json').read_text())['split_sha256']
    model=build_model('beetleformer');model.load_state_dict(ck['model']);model.eval()
    rng=np.random.default_rng(CONFIG['seed'])
    pos=rng.choice(np.flatnonzero(df.partition.eq('test')&df.label.eq(1)),750,replace=False)
    neg=rng.choice(np.flatnonzero(df.partition.eq('test')&df.label.eq(0)),750,replace=False)
    subset=np.concatenate([pos,neg]);seq=df.seq.tolist()
    basepref=[]
    for j in range(60):
        basepref.append(dict(position=j+1,**{b:float(np.mean([seq[i][j]==b for i in pos])-np.mean([seq[i][j]==b for i in neg])) for b in 'ACGT'}))
    pd.DataFrame(basepref).to_csv(RESULTS/'interpretation_base_preference.csv',index=False)
    zs=[]
    with torch.no_grad():
        for start in range(0,len(subset),32):
            ix=subset[start:start+32];zs.append(model(ids[ix],mask[ix],feat[ix])['embedding'].numpy())
    pca=PCA(n_components=2,svd_solver='full');coords=pca.fit_transform(np.concatenate(zs))
    pd.DataFrame(dict(id=df.iloc[subset].id,label=df.iloc[subset].label,PC1=coords[:,0],PC2=coords[:,1])).to_csv(RESULTS/'interpretation_pca.csv',index=False)
    saliency=[]; sample=pos[:200]; captured={}
    def hook(module,inp,out):
        captured['embedding']=out;out.retain_grad()
    handle=model.embedding.register_forward_hook(hook)
    for start in range(0,len(sample),16):
        ix=sample[start:start+16];n=len(ix);model.zero_grad(set_to_none=True)
        model(ids[ix],mask[ix],feat[ix])['lethality_logit'].sum().backward()
        g=captured['embedding'].grad.norm(dim=2).detach().numpy()
        for j,row in enumerate(ix):
            length=len(seq[row]);v=g[j,:length]+g[j+n,:length][::-1]
            v=v/max(v.sum(),1e-12);v=np.interp(np.linspace(0,1,50),np.linspace(0,1,length),v);v=v/max(v.sum(),1e-12)
            saliency.append(v)
        print('saliency',min(start+16,len(sample)),'/',len(sample),flush=True)
    handle.remove();values=np.stack(saliency)
    pd.DataFrame(dict(relative_position=np.linspace(0,1,50),mean=values.mean(0),sd=values.std(0))).to_csv(RESULTS/'interpretation_saliency.csv',index=False)
    pd.DataFrame(dict(id=df.iloc[sample].id)).to_csv(RESULTS/'saliency_sample_ids.csv',index=False)
    (RESULTS/'interpretation_metadata.json').write_text(json.dumps(dict(seed=CONFIG['seed'],base_preference_n_each=750,
        saliency_n=200,PCA_n=1500,PCA_explained_variance=pca.explained_variance_ratio_.tolist(),
        saliency='L2 norm of logit gradient on embeddings, two orientations aligned and added, normalized per sequence then interpolated to 50 relative-position bins and normalized again',
        checkpoint_epoch=ck['epoch']),indent=2))


if __name__=='__main__':main()
