#!/bin/bash
# FAST binary comparison: tiny dim, few steps, conv_stem on, big batch.
set -e
PY=/Users/flash/opt/anaconda3/envs/pyg/bin/python
cd /Users/flash/Desktop/AI+RNAi
BASE="device=cpu model.dim=32 model.depth=2 model.heads=4 model.stem_pool=8 train.batch_size=128 train.total_steps=200 train.warmup_steps=20 train.eval_every=200 train.log_every=100 train.num_workers=2 paths.out_dir=logs/bin"

run () { local name=$1; shift
  echo "########## $name ##########"
  $PY scripts/train_binary.py --config config/rnai.yaml --set $BASE "$@" \
      run_name=$name paths.ckpt_dir=checkpoint/$name 2>&1 \
    | grep -vE "UserWarning|warnings.warn|enable_nested" | grep -E "binary:|final" ; }

run bf_stem  model.arch=transformer model.conv_stem=true
run bf_feat  model.arch=transformer model.conv_stem=true data.use_features=true
run bf_kmer  model.arch=transformer model.conv_stem=true data.kmer=3
run bf_cnn   model.arch=cnn_bilstm model.dim=128
echo "########## ALL DONE ##########"
