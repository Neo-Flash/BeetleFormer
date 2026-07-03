#!/bin/bash
# Fast 2x2 matrix: {pooled, short 130-399, long 400-616} x {single-task dali11, multi-task dali11+5pheno}
# plus CNN-BiLSTM baseline and GC/length ridge lower bound. Small config, quick test.
set -e
PY=/Users/flash/opt/anaconda3/envs/pyg/bin/python
cd /Users/flash/Desktop/AI+RNAi
PHENO="pupal_phase_1_stage_1_cuticle,pupal_phase_1_metamorphosis,larval_phase_1_metamorphosis,pupal_phase_1_musculature,larval_phase_1_fertility"
# tiny + fast: dim64 depth2, 400 steps, single final eval
BASE="device=cpu data.pheno_topics=$PHENO model.arch=transformer model.dim=64 model.depth=2 model.heads=4 model.conv_stem=true model.stem_pool=4 train.batch_size=64 train.total_steps=400 train.warmup_steps=40 train.eval_every=400 train.log_every=200 train.num_workers=2 paths.out_dir=logs/exp"

run () { # name  extra_sets...
  local name=$1; shift
  echo "########## $name ##########"
  $PY scripts/train.py --config config/rnai.yaml --set $BASE "$@" \
      run_name=$name paths.ckpt_dir=checkpoint/$name 2>&1 \
    | grep -vE "UserWarning|warnings.warn|enable_nested" | grep -E "length group|subset|dsRNA:|final"
}

# --- pooled ---
run m_pool_solo  train.pheno_weight=0
run m_pool_multi train.pheno_weight=1
# --- short group 130-399 ---
run m_short_solo  data.len_min=130 data.len_max=399 train.pheno_weight=0
run m_short_multi data.len_min=130 data.len_max=399 train.pheno_weight=1
# --- long group 400-616 ---
run m_long_solo  data.len_min=400 data.len_max=616 train.pheno_weight=0
run m_long_multi data.len_min=400 data.len_max=616 train.pheno_weight=1

# --- baselines ---
echo "########## baseline CNN-BiLSTM (pooled, multi) ##########"
$PY scripts/train.py --config config/rnai.yaml --set $BASE \
    model.arch=cnn_bilstm model.dim=128 train.pheno_weight=1 \
    run_name=m_cnn paths.ckpt_dir=checkpoint/m_cnn 2>&1 \
  | grep -vE "UserWarning|warnings.warn|enable_nested" | grep -E "dsRNA:|final"

echo "########## baseline GC/length ridge (pooled) ##########"
$PY scripts/evaluate.py --config config/rnai.yaml --set $BASE --baseline-only 2>&1 \
  | grep -vE "UserWarning|warnings.warn|enable_nested" | grep -iE "ridge|spearman"

echo "########## ALL DONE ##########"
