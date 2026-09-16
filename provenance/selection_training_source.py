"""Prospectively fixed source-gene-disjoint sensitivity analysis (seed 20260911).

Uses the unchanged historical sequence benchmark, restored negative provenance,
and identical optimization/input features for both newly fitted neural models.
The held-out test is evaluated only after validation-based epoch selection.
This does not turn unlabelled transcript fragments into biological negatives.
"""
from pathlib import Path
import sys, json, random, time, hashlib, argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, matthews_corrcoef, confusion_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_binary import build_model
from utils.config import Cfg
from data.rnai_dataset import encode_seq
from data.features import compute_features, FEATURE_NAMES
from revision_20260911.model_revision import MaskedMultiScaleTransformer
OUT=ROOT/'revision_20260911/experiments';OUT.mkdir(exist_ok=True)
SEED=20260911

def report(y,p):
 tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel()
 return dict(n=len(y),AUC=float(roc_auc_score(y,p)),AP=float(average_precision_score(y,p)),Acc=float(accuracy_score(y,p>=.5)),MCC=float(matthews_corrcoef(y,p>=.5)),TP=int(tp),TN=int(tn),FP=int(fp),FN=int(fn))

def prepare():
 df=pd.read_csv(ROOT/'data/ibeetle_rnai/manifest_binary.csv').fillna('')
 prov=pd.read_csv(ROOT/'revision_20260911/audit/negative_source_provenance.csv').set_index('id')
 df['source_gene']=[prov.loc[r.id,'source_gene'] if r.label==0 else (r.gene or 'unknown_'+r.id) for r in df.itertuples()]
 # Link genes sharing an exact sequence (including its reverse complement).
 parent={g:g for g in df.source_gene.unique()}
 def find(g):
  while parent[g]!=g:parent[g]=parent[parent[g]];g=parent[g]
  return g
 first={};trans=str.maketrans('ACGT','TGCA')
 for r in df.itertuples():
  key=min(r.seq,r.seq.translate(trans)[::-1]);g=r.source_gene
  if key in first:parent[find(g)]=find(first[key])
  else:first[key]=g
 df['group']=[find(g) for g in df.source_gene]
 groups=df.group.to_numpy();y=df.label.to_numpy()
 trva,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(df,y,groups))
 a,b=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=SEED+1).split(df.iloc[trva],y[trva],groups[trva]))
 tr,va=trva[a],trva[b]
 df['partition']='';df.loc[tr,'partition']='train';df.loc[va,'partition']='validation';df.loc[te,'partition']='test'
 df.to_csv(OUT/'source_gene_split.csv',index=False)
 counts={s:{'n':int((df.partition==s).sum()),'positive':int(df.loc[df.partition==s,'label'].sum()),'source_genes':int(df.loc[df.partition==s,'source_gene'].nunique())} for s in ['train','validation','test']}
 for a,b in [('train','test'),('train','validation'),('validation','test')]:
  assert not set(df.loc[df.partition==a,'source_gene'])&set(df.loc[df.partition==b,'source_gene'])
 protocol=dict(seed=SEED,epochs=12,batch_size=128,optimizer='AdamW',learning_rate=.0003,weight_decay=.01,warmup_fraction=.1,selection='maximum validation AUC; first epoch wins ties; architecture also selected on validation AUC',test_use='neural test outputs generated only after all candidate training completes; no test tuning',candidate_models=['transformer','cnn_faithful','masked_multiscale','strand_symmetric'],features=list(FEATURE_NAMES),symmetric_features='average of descriptors from forward and reverse-complement strands',revision_note='architecture candidates fixed following user instruction before any neural test evaluation; interrupted original training restarted because optimizer was not checkpointed',split=counts,sha256=hashlib.sha256((OUT/'source_gene_split.csv').read_bytes()).hexdigest())
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 print('PREDEFINED PROTOCOL',json.dumps(protocol),flush=True)
 if not (OUT/'inputs.npz').exists():
  encoded=[encode_seq(s,640,1) for s in df.seq]
  ids=torch.stack([x[0] for x in encoded]).numpy();attn=torch.stack([x[1] for x in encoded]).numpy()
  feat=np.stack([compute_features(s,640) for s in df.seq]);np.savez_compressed(OUT/'inputs.npz',ids=ids,attn=attn,feat=feat,y=y)
 return df,tr,va,te

@torch.no_grad()
def predict(model,ids,attn,feat,idx,device):
 model.eval();ps=[]
 for start in range(0,len(idx),128):
  ix=idx[start:start+128]
  out=model(ids[ix].to(device),attn[ix].to(device),feat[ix].to(device))
  ps.extend(torch.sigmoid(out['lethality_logit'][:,0]).cpu().numpy().tolist())
 return np.array(ps)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=['transformer','cnn_faithful','masked_multiscale','strand_symmetric']);args=ap.parse_args()
 torch.set_num_threads(6);df,tr,va,te=prepare();a=np.load(OUT/'inputs.npz');y=a['y'];feat=a['feat']
 for name,cols in [('length_only',[5]),('composition_length',list(range(6))),('features_only',list(range(feat.shape[1])))]:
  clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=3000,random_state=SEED)).fit(feat[tr][:,cols],y[tr]);p=clf.predict_proba(feat[te][:,cols])[:,1]
  pd.DataFrame({'id':df.iloc[te].id,'label':y[te],'score':p}).to_csv(OUT/(name+'_test.csv'),index=False)
  (OUT/(name+'_metrics.json')).write_text(json.dumps(report(y[te],p),indent=2))
 if not (OUT/'features_rc.npy').exists():
  comp=str.maketrans('ACGT','TGCA');np.save(OUT/'features_rc.npy',np.stack([compute_features(s.translate(comp)[::-1],640) for s in df.seq]))
 invariant_ft=torch.from_numpy((feat+np.load(OUT/'features_rc.npy'))/2)
 ids=torch.from_numpy(a['ids']);attn=torch.from_numpy(a['attn']);base_ft=torch.from_numpy(feat);yt=torch.from_numpy(y.astype(np.float32));device='mps' if torch.backends.mps.is_available() else 'cpu'
 def construct(arch,cfg):
  return MaskedMultiScaleTransformer(symmetric=arch=='strand_symmetric') if arch in ['masked_multiscale','strand_symmetric'] else build_model(cfg)
 for arch in args.models:
  if (OUT/(arch+'_training_complete.json')).exists():continue
  ft=invariant_ft if arch=='strand_symmetric' else base_ft
  torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
  cfg=Cfg({'data':{'max_len':640,'kmer':1,'use_features':True},'model':{'arch':arch,'dim':192 if arch=='transformer' else 100,'depth':4,'heads':6,'mlp_ratio':4,'dropout':.1,'conv_stem':True,'stem_kernel':9,'stem_layers':2,'stem_pool':4,'pool':'mean'}})
  model=construct(arch,cfg).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
  batches=int(np.ceil(len(tr)/128));total=12*batches;step=0;best=-1;history=[];rng=np.random.default_rng(SEED);started=time.time()
  first_epoch=1;resume=OUT/(arch+'_resume.pt')
  if resume.exists():
   rr=torch.load(resume,map_location='cpu',weights_only=False);model.load_state_dict(rr['model']);opt.load_state_dict(rr['optimizer']);step=rr['step'];best=rr['best'];history=rr['history'];first_epoch=rr['epoch']+1;rng.bit_generator.state=rr['numpy_rng'];torch.set_rng_state(rr['torch_rng'])
   if device=='mps':torch.mps.set_rng_state(rr['mps_rng'])
   print('RESUME',arch,'epoch',first_epoch,flush=True)
  print('TRAIN',arch,device,'params',sum(p.numel() for p in model.parameters()),flush=True)
  for epoch in range(first_epoch,13):
   model.train();order=rng.permutation(tr);loss_sum=0
   for start in range(0,len(order),128):
    ix=order[start:start+128];step+=1;warm=int(.1*total)
    lr=.0003*(step/max(1,warm) if step<=warm else .5*(1+np.cos(np.pi*(step-warm)/(total-warm))))
    for g in opt.param_groups:g['lr']=lr
    out=model(ids[ix].to(device),attn[ix].to(device),ft[ix].to(device));loss=torch.nn.functional.binary_cross_entropy_with_logits(out['lethality_logit'][:,0],yt[ix].to(device))
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();loss_sum+=loss.item()*len(ix)
    if step%25==0:print(arch,'epoch',epoch,'step',step,'/',total,'loss',round(loss.item(),4),'seconds',round(time.time()-started),flush=True)
   p=predict(model,ids,attn,ft,va,device);r=report(y[va],p);r.update(epoch=epoch,loss=loss_sum/len(tr));history.append(r)
   print('VALIDATION',arch,json.dumps(r),flush=True)
   if r['AUC']>best:
    best=r['AUC'];torch.save({'model':{k:v.cpu() for k,v in model.state_dict().items()},'cfg':dict(cfg),'epoch':epoch,'validation':r},OUT/(arch+'_best.pt'))
   (OUT/(arch+'_history.json')).write_text(json.dumps(history,indent=2))
   torch.save({'model':{k:v.cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'step':step,'best':best,'history':history,'epoch':epoch,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'mps_rng':torch.mps.get_rng_state() if device=='mps' else None},resume)
  (OUT/(arch+'_training_complete.json')).write_text(json.dumps({'seconds':time.time()-started,'best_validation_AUC':best,'params':sum(p.numel() for p in model.parameters())},indent=2))
 # Freeze candidate choice before any neural test inference.
 records={arch:json.load(open(OUT/(arch+'_training_complete.json'))) for arch in args.models}
 selected=max(records,key=lambda k:records[k]['best_validation_AUC'])
 (OUT/'selected_architecture.json').write_text(json.dumps({'selected':selected,'criterion':'maximum validation AUC','candidates':records},indent=2))
 for arch in args.models:
  if (OUT/(arch+'_metrics.json')).exists():continue
  ft=invariant_ft if arch=='strand_symmetric' else base_ft
  ck=torch.load(OUT/(arch+'_best.pt'),weights_only=False,map_location='cpu');model=construct(arch,Cfg(ck['cfg'])).to(device);model.load_state_dict(ck['model']);p=predict(model,ids,attn,ft,te,device);best=records[arch]['best_validation_AUC']
  r=report(y[te],p);r.update(selected_epoch=ck['epoch'],validation_AUC=best,params=sum(p.numel() for p in model.parameters()),seconds=records[arch]['seconds'])
  pd.DataFrame({'id':df.iloc[te].id,'label':y[te],'score':p}).to_csv(OUT/(arch+'_test.csv'),index=False)
  (OUT/(arch+'_metrics.json')).write_text(json.dumps(r,indent=2));print('FINAL TEST',arch,json.dumps(r),flush=True)

if __name__=='__main__':main()
