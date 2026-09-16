"""Create a fresh, self-contained reproduction directory without overwriting files."""
from pathlib import Path
import argparse,shutil,subprocess,sys,json,hashlib,os
from common import ROOT,CONFIG

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--retrain',action='store_true',help='Train seven primary configurations and two gene-isolated models from scratch')
    ap.add_argument('--device',choices=['cpu','mps','cuda'])
    ap.add_argument('--prepare-only',action='store_true',help='Rebuild and check data, then stop')
    args=ap.parse_args();dest=args.output_dir.expanduser().resolve()
    if dest.exists():raise ValueError('Output directory must not exist: '+str(dest))
    if ROOT in dest.parents:raise ValueError('Choose an output directory outside the source code directory')
    code=dest/'code';code.mkdir(parents=True)
    for name in ['README.md','requirements.txt']:
        shutil.copy2(ROOT/name,code/name)
    for name in ['scripts','model','utils','config','provenance']:
        shutil.copytree(ROOT/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'data',code/'data',ignore=shutil.ignore_patterns('inputs.npz','features_rc.npy','__pycache__','*.pyc','cedden'))
    (code/'results').mkdir();(code/'checkpoints').mkdir();(code/'logs').mkdir()
    if not (args.retrain or args.prepare_only):
        for name in CONFIG['models']:
            shutil.copy2(ROOT/'checkpoints'/(name+'_best.pt'),code/'checkpoints'/(name+'_best.pt'))
            for suffix in ['_history.json','_training_complete.json','_metrics.json','_test.csv']:
                shutil.copy2(ROOT/'results'/(name+suffix),code/'results'/(name+suffix))
        for folder in ['results','checkpoints']:(code/'strict_gene'/folder).mkdir(parents=True,exist_ok=True)
        for name in ['beetleformer','cnn_bilstm']:
            shutil.copy2(ROOT/'strict_gene/checkpoints'/(name+'_best.pt'),code/'strict_gene/checkpoints'/(name+'_best.pt'))
            for suffix in ['_history.json','_training_complete.json','_metrics.json','_test.csv']:
                shutil.copy2(ROOT/'strict_gene/results'/(name+suffix),code/'strict_gene/results'/(name+suffix))
    calls=[]
    def run(script,*extra,protocol='sequence'):
        cmd=[sys.executable,str(code/'scripts'/script),*extra]
        env=dict(os.environ,BEETLEFORMER_RUN_ROOT=str(code if protocol=='sequence' else code/'strict_gene'))
        print('RUN',protocol,script,*extra,flush=True);subprocess.run(cmd,cwd=code,env=env,check=True);calls.append(dict(protocol=protocol,command=cmd))
    run('prepare_data.py','--rebuild-cache')
    run('audit_protocols.py')
    run('audit_original_sources.py')
    if not args.prepare_only:
        if args.retrain:run('train_two_protocols.py',*(['--device',args.device] if args.device else []))
        run('analyze_results.py');run('external_validation.py');run('external_validation.py',protocol='gene');run('interpret_model.py')
        run('make_figures.py');run('check_consistency.py');run('check_consistency.py',protocol='gene')
    manifest={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    (dest/'reproduction_manifest.json').write_text(json.dumps(dict(mode='retrain' if args.retrain else 'released checkpoints',commands=calls,sha256=manifest),indent=2))
    print('Reproduction output:',dest)

if __name__=='__main__':main()
