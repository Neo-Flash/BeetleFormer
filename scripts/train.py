"""Fit models with one frozen source-gene split and shared optimization protocol.

Example: python scripts/train.py --models without_convolutions without_descriptors
Existing completed models are not overwritten. Use --output-root for a fresh run.
All requested models finish training before their held-out test is evaluated.
"""
from pathlib import Path
import argparse, hashlib, json, random, time
import numpy as np
import pandas as pd
import torch
from common import ROOT, RUN_ROOT, RUN_DATA, RESULTS, CHECKPOINTS, CONFIG, build_model, report, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', choices=list(CONFIG['models']), default=list(CONFIG['models']))
    ap.add_argument('--output-root', type=Path)
    ap.add_argument('--device', choices=['mps', 'cpu', 'cuda'])
    args = ap.parse_args()
    out = args.output_root / 'results' if args.output_root else RESULTS
    ckdir = args.output_root / 'checkpoints' if args.output_root else CHECKPOINTS
    out.mkdir(exist_ok=True, parents=True); ckdir.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(CONFIG['cpu_threads'])
    seed = CONFIG['seed']; batch = CONFIG['batch_size']; epochs = CONFIG['epochs']
    df = pd.read_csv(RESULTS / 'source_gene_split.csv').fillna('')
    indices = {k: np.flatnonzero(df.partition.eq(k)) for k in ['train', 'validation', 'test']}
    for a, b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(df.iloc[indices[a]].source_gene) & set(df.iloc[indices[b]].source_gene)
        assert not set(df.iloc[indices[a]].group) & set(df.iloc[indices[b]].group)
    checks=json.loads((RESULTS/'data_checks.json').read_text())
    isolation=json.loads((RESULTS/'external_isolation_checks.json').read_text())
    assert isolation['sequence_matches']==0
    if checks['protocol']=='gene':assert not isolation['genes_shared']
    fingerprint=hashlib.sha256((RESULTS/'source_gene_split.csv').read_bytes()).hexdigest()
    assert fingerprint==checks['split_sha256']
    cache = np.load(RUN_DATA / 'inputs.npz')
    assert hashlib.sha256((RUN_DATA/'inputs.npz').read_bytes()).hexdigest()==checks['dataset_sha256']
    assert np.array_equal(cache['y'], df.label.to_numpy())
    ids = torch.from_numpy(cache['ids']); mask = torch.from_numpy(cache['attn'])
    base = torch.from_numpy(cache['feat'])
    symmetric = torch.from_numpy((cache['feat'] + np.load(RUN_DATA / 'features_rc.npy')) / 2)
    y = cache['y']; yt = torch.from_numpy(y.astype(np.float32))
    device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    tr, va, te = (indices[k] for k in ['train','validation','test'])
    frozen = dict(configuration=CONFIG, requested_models=args.models,
                  split_sha256=hashlib.sha256((RESULTS/'source_gene_split.csv').read_bytes()).hexdigest(),
                  created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  evaluation_protocol=checks['protocol'],external_identity_sha256=checks['external_identity_sha256'],
                  note='Architecture and optimizer fixed before this two-protocol retraining; external outcomes are never used for checkpoint selection or retuning.')
    (out / 'ablation_protocol.json').write_text(json.dumps(frozen, indent=2))
    for name in args.models:
        if (out / (name + '_training_complete.json')).exists():
            prior=json.loads((out/(name+'_training_complete.json')).read_text())
            assert prior.get('split_sha256')==fingerprint,'Refusing to reuse a model trained on another split'
            print('REUSE completed', name, flush=True); continue
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        ft = symmetric if CONFIG['models'][name]['symmetric_features'] else base
        model = build_model(name).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        batches = int(np.ceil(len(tr)/batch)); total = epochs*batches
        step = 0; best = -1; history = []; first_epoch = 1
        rng = np.random.default_rng(seed); started = time.time(); elapsed_before = 0
        resume = ckdir / (name + '_resume.pt')
        if resume.exists():
            rr = torch.load(resume, map_location='cpu', weights_only=False)
            assert rr.get('split_sha256')==fingerprint,'Resume checkpoint belongs to another split'
            model.load_state_dict(rr['model']); opt.load_state_dict(rr['optimizer'])
            step = rr['step']; best = rr['best']; history = rr['history']; first_epoch = rr['epoch']+1
            rng.bit_generator.state = rr['numpy_rng']; torch.set_rng_state(rr['torch_rng'])
            if device == 'mps': torch.mps.set_rng_state(rr['mps_rng'])
            elapsed_before = rr['elapsed_seconds']
        print('TRAIN', name, device, 'params', sum(p.numel() for p in model.parameters()), flush=True)
        for epoch in range(first_epoch, epochs+1):
            model.train(); order = rng.permutation(tr); loss_sum = 0
            for start in range(0, len(order), batch):
                ix = order[start:start+batch]; step += 1; warm = int(CONFIG['warmup_fraction']*total)
                lr = CONFIG['learning_rate']*(step/max(1,warm) if step <= warm else .5*(1+np.cos(np.pi*(step-warm)/(total-warm))))
                for group in opt.param_groups: group['lr'] = lr
                pred = model(ids[ix].to(device), mask[ix].to(device), ft[ix].to(device))['lethality_logit'][:,0]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, yt[ix].to(device))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip']); opt.step()
                loss_sum += loss.item()*len(ix)
                if step % 50 == 0:
                    print(name, 'epoch', epoch, 'step', step, '/', total, 'loss', round(loss.item(),4), flush=True)
            p = predict(model, ids, mask, ft, va, device)
            r = report(y[va], p); r.update(epoch=epoch, loss=loss_sum/len(tr)); history.append(r)
            print('VALIDATION', name, json.dumps(r), flush=True)
            if r['AUC'] > best:
                best = r['AUC']
                torch.save({'model':{k:v.cpu() for k,v in model.state_dict().items()}, 'model_name':name,
                            'config':CONFIG, 'epoch':epoch, 'validation':r,'split_sha256':fingerprint,'evaluation_protocol':checks['protocol']}, ckdir/(name+'_best.pt'))
            (out/(name+'_history.json')).write_text(json.dumps(history,indent=2))
            torch.save({'model':{k:v.cpu() for k,v in model.state_dict().items()}, 'optimizer':opt.state_dict(),
                        'step':step, 'best':best, 'history':history, 'epoch':epoch,
                        'numpy_rng':rng.bit_generator.state, 'torch_rng':torch.get_rng_state(),
                        'mps_rng':torch.mps.get_rng_state() if device=='mps' else None,
                        'elapsed_seconds':elapsed_before+time.time()-started,'split_sha256':fingerprint}, resume)
        record = dict(seconds=elapsed_before+time.time()-started, best_validation_AUC=best,
                      params=sum(p.numel() for p in model.parameters()), device=device, steps=step,split_sha256=fingerprint,evaluation_protocol=checks['protocol'])
        (out/(name+'_training_complete.json')).write_text(json.dumps(record,indent=2))
    # No test-dependent epoch choice: every checkpoint above was selected on validation only.
    for name in args.models:
        if (out/(name+'_metrics.json')).exists(): continue
        ft = symmetric if CONFIG['models'][name]['symmetric_features'] else base
        ck = torch.load(ckdir/(name+'_best.pt'), map_location='cpu', weights_only=False)
        model = build_model(name).to(device); model.load_state_dict(ck['model'])
        p = predict(model, ids, mask, ft, te, device)
        r = report(y[te],p)
        r.update(selected_epoch=ck['epoch'], validation_AUC=ck['validation']['AUC'],
                 params=sum(p.numel() for p in model.parameters()))
        pd.DataFrame({'id':df.iloc[te].id, 'label':y[te], 'score':p}).to_csv(out/(name+'_test.csv'),index=False)
        (out/(name+'_metrics.json')).write_text(json.dumps(r,indent=2))
        print('TEST', name, json.dumps(r),flush=True)


if __name__ == '__main__': main()
