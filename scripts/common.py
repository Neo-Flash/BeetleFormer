"""Shared paths, configuration, model construction and evaluation."""
from pathlib import Path
import sys, json, os
import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, matthews_corrcoef, confusion_matrix)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN_ROOT = Path(os.environ.get('BEETLEFORMER_RUN_ROOT', str(ROOT))).resolve()
RESULTS = RUN_ROOT / 'results'
CHECKPOINTS = RUN_ROOT / 'checkpoints'
RUN_DATA = RUN_ROOT / 'data'
CONFIG = json.loads((ROOT / 'config/experiment.json').read_text())
from model.beetleformer import BeetleFormer
from model.seq_transformer import dsRNATransformer
from model.baselines import AdaptedCNNBiLSTM


def build_model(name):
    if name == 'cnn_bilstm':
        # An adapted CNN-BiLSTM comparator, not the published dsRNAPredictor implementation.
        return AdaptedCNNBiLSTM(dim=100, num_lethality=1, num_topics=1,
                                vocab_size=6, extra_dim=30)
    if name == 'single_scale_transformer':
        return dsRNATransformer(dim=192, depth=4, heads=6, mlp_ratio=4,
                               dropout=.1, max_len=640, num_lethality=1,
                               num_topics=1, pool='mean', conv_stem=True,
                               stem_kernel=9, stem_layers=2, stem_pool=4,
                               vocab_size=6, extra_dim=30)
    return BeetleFormer(**CONFIG['models'][name]['parameters'])


def report(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p >= .5, labels=[0, 1]).ravel()
    return dict(n=len(y), AUC=float(roc_auc_score(y, p)),
                AP=float(average_precision_score(y, p)),
                Acc=float(accuracy_score(y, p >= .5)),
                MCC=float(matthews_corrcoef(y, p >= .5)),
                Sn=float(tp/(tp+fn)), Sp=float(tn/(tn+fp)),
                Precision=float(tp/max(1, tp+fp)), F1=float(2*tp/max(1, 2*tp+fp+fn)),
                TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn))


@torch.no_grad()
def predict(model, ids, mask, feat, idx, device, logits=False):
    model.eval()
    ps = []
    for start in range(0, len(idx), CONFIG['batch_size']):
        ix = idx[start:start+CONFIG['batch_size']]
        out = model(ids[ix].to(device), mask[ix].to(device), feat[ix].to(device))
        values = out['lethality_logit'][:, 0]
        ps.extend((values if logits else torch.sigmoid(values)).cpu().numpy().tolist())
    return np.array(ps)
