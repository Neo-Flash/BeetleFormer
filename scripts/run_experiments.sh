#!/bin/bash
set -e
PY=/Users/flash/opt/anaconda3/envs/pyg/bin/python
cd /Users/flash/Desktop/AI+RNAi
PHENO="pupal_phase_1_stage_1_cuticle,pupal_phase_1_metamorphosis,larval_phase_1_metamorphosis,pupal_phase_1_musculature,larval_phase_1_fertility"
COMMON="device=cpu data.pheno_topics=$PHENO model.dim=96 model.depth=2 model.heads=4 train.batch_size=32 train.total_steps=1000 train.warmup_steps=80 train.eval_every=500 train.log_every=250 train.num_workers=2 paths.out_dir=logs/exp"

echo "########## [1/4] Transformer + conv_stem, MULTI-TASK (dali11 + 5 pheno) ##########"
$PY scripts/train.py --config config/rnai.yaml --set $COMMON \
  model.arch=transformer model.conv_stem=true model.stem_pool=4 \
  run_name=exp_tf_multi paths.ckpt_dir=checkpoint/exp_tf_multi 2>&1 | grep -vE "UserWarning|warnings.warn|enable_nested"

echo "########## [2/4] Transformer + conv_stem, LETHALITY-ONLY (pheno_weight=0) ##########"
$PY scripts/train.py --config config/rnai.yaml --set $COMMON \
  model.arch=transformer model.conv_stem=true model.stem_pool=4 train.pheno_weight=0 \
  run_name=exp_tf_solo paths.ckpt_dir=checkpoint/exp_tf_solo 2>&1 | grep -vE "UserWarning|warnings.warn|enable_nested"

echo "########## [3/4] CNN-BiLSTM baseline, MULTI-TASK ##########"
$PY scripts/train.py --config config/rnai.yaml --set $COMMON \
  model.arch=cnn_bilstm model.dim=128 \
  run_name=exp_cnn paths.ckpt_dir=checkpoint/exp_cnn 2>&1 | grep -vE "UserWarning|warnings.warn|enable_nested"

echo "########## [4/4] GC/length ridge lower bound ##########"
$PY scripts/evaluate.py --config config/rnai.yaml --set $COMMON --baseline-only 2>&1 | grep -vE "UserWarning|warnings.warn|enable_nested"

echo "########## ALL DONE ##########"
