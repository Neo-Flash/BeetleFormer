#!/bin/bash
# Compare improvement techniques on pooled multi-task (dali11 + 5 pheno).
# base vs +thermo/tiling features vs +ranking loss vs +3-mer vs all combined.
set -e
PY=/Users/flash/opt/anaconda3/envs/pyg/bin/python
cd /Users/flash/Desktop/AI+RNAi
PHENO="pupal_phase_1_stage_1_cuticle,pupal_phase_1_metamorphosis,larval_phase_1_metamorphosis,pupal_phase_1_musculature,larval_phase_1_fertility"
BASE="device=cpu data.pheno_topics=$PHENO model.arch=transformer model.dim=64 model.depth=2 model.heads=4 model.conv_stem=true model.stem_pool=4 train.batch_size=64 train.total_steps=400 train.warmup_steps=40 train.eval_every=400 train.log_every=200 train.num_workers=2 train.pheno_weight=1 paths.out_dir=logs/imp"

run () { local name=$1; shift
  echo "########## $name ##########"
  $PY scripts/train.py --config config/rnai.yaml --set $BASE "$@" \
      run_name=$name paths.ckpt_dir=checkpoint/$name 2>&1 \
    | grep -vE "UserWarning|warnings.warn|enable_nested" | grep -E "dsRNA:|final" ; }

run i_base
run i_feat  data.use_features=true
run i_rank  train.rank_weight=0.5
run i_kmer  data.kmer=3
run i_all   data.use_features=true train.rank_weight=0.5 data.kmer=3
echo "########## ALL DONE ##########"
