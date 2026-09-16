"""Run the frozen sequence-isolated benchmark and gene-isolated sensitivity test."""
import os, subprocess, sys, argparse
from common import ROOT

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','mps','cuda']);args=ap.parse_args()
    for protocol, run_root, models in [
        ('sequence', ROOT, []),
        ('gene', ROOT/'strict_gene', ['--models','beetleformer','cnn_bilstm']),
    ]:
        env=dict(os.environ, BEETLEFORMER_RUN_ROOT=str(run_root))
        print('START PROTOCOL',protocol,flush=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/train.py'),*models,*(['--device',args.device] if args.device else [])],env=env,cwd=ROOT,check=True)
        print('COMPLETE PROTOCOL',protocol,flush=True)
