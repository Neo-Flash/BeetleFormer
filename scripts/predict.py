"""Score FASTA sequences with the frozen BeetleFormer checkpoint."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from common import CONFIG,CHECKPOINTS,build_model,predict
from data.rnai_dataset import encode_seq
from data.features import compute_features


def read_fasta(path):
    records=[];name=None;parts=[]
    for line in path.read_text().splitlines():
        if line.startswith('>'):
            if name is not None:records.append((name,''.join(parts)))
            name=line[1:].strip();parts=[]
        elif line.strip():
            if name is None:raise ValueError('Sequence before FASTA header')
            parts.append(line.strip().upper().replace('U','T'))
    if name is not None:records.append((name,''.join(parts)))
    if not records:raise ValueError('No FASTA records')
    for name,seq in records:
        if not name or set(seq)-set('ACGT'):raise ValueError(f'{name}: use unambiguous A/C/G/T/U nucleotides')
        if not 130<=len(seq)<=615:raise ValueError(f'{name}: length {len(seq)} outside evaluated range 130–615')
    return records


def main():
    ap=argparse.ArgumentParser();ap.add_argument('fasta',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,default=CHECKPOINTS/'beetleformer_best.pt');args=ap.parse_args()
    torch.set_num_threads(3);records=read_fasta(args.fasta);seqs=[s for _,s in records]
    enc=[encode_seq(s,CONFIG['max_len'],1) for s in seqs];trans=str.maketrans('ACGT','TGCA')
    features=torch.from_numpy(np.stack([(compute_features(s,640)+compute_features(s.translate(trans)[::-1],640))/2 for s in seqs]))
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False);model=build_model('beetleformer');model.load_state_dict(ck['model'])
    logits=predict(model,torch.stack([e[0] for e in enc]),torch.stack([e[1] for e in enc]),features,np.arange(len(records)),'cpu',logits=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(dict(id=[n for n,_ in records],length=[len(s) for s in seqs],logit=logits,
                      score=1/(1+np.exp(-logits)))).to_csv(args.output,index=False)
    print(f'Scored {len(records)} sequences. Scores measure screen-positive/background discrimination, not calibrated knockdown.')


if __name__=='__main__':main()
