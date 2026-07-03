#!/bin/bash
# Binary dsRNA-vs-mRNA (dsRNAPredictor-style). Compare plain baseline vs
# improvement knobs vs CNN-BiLSTM. Random stratified 80/20. Small fast config.
set -e
PY=/Users/flash/opt/anaconda3/envs/pyg/bin/python
cd /Users/flash/Desktop/AI+RNAi
BASE="device=cpu model.dim=64 model.depth=2 model.heads=4 model.stem_pool=4 train.batch_size=64 train.total_steps=600 train.warmup_steps=60 train.eval_every=600 train.log_every=300 train.num_workers=2 paths.out_dir=logs/bin"

run () { local name=$1; shift
  echo "########## $name ##########"
  $PY scripts/train_binary.py --config config/rnai.yaml --set $BASE "$@" \
      run_name=$name paths.ckpt_dir=checkpoint/$name 2>&1 \
    | grep -vE "UserWarning|warnings.warn|enable_nested" | grep -E "binary:|final" ; }

run b_base   model.arch=transformer model.conv_stem=false
run b_stem   model.arch=transformer model.conv_stem=true
run b_feat   model.arch=transformer model.conv_stem=true data.use_features=true
run b_kmer   model.arch=transformer model.conv_stem=true data.kmer=3
run b_cnn    model.arch=cnn_bilstm model.dim=128
echo "########## ALL DONE ##########"
