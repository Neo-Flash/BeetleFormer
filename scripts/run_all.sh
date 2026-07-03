#!/bin/bash
# =============================================================================
# run_all.sh — one-click runner for ALL dsRNATransformer binary experiments.
#
#   Task: binary classification, effective dsRNA vs mRNA fragment
#         (mirrors dsRNAPredictor / dsRNAPredictor-II).
#
#   Usage:
#     bash scripts/run_all.sh                 # full suite on GPU (device=cuda)
#     DEVICE=cpu bash scripts/run_all.sh      # force CPU
#     STEPS=5000 bash scripts/run_all.sh      # override training steps
#     ONLY=core bash scripts/run_all.sh       # only the core subset (2 + 3.2 + 4.1)
#
#   Prints a live progress banner before each experiment: which one is running,
#   how many are done / left, and elapsed time.
# =============================================================================
set -u
PY=${PY:-python}                # use the active env's python; override with PY=...
DEVICE=${DEVICE:-cuda}          # GPU by default; single-card, no index needed
STEPS=${STEPS:-8000}
ONLY=${ONLY:-all}               # all | core
cd "$(dirname "$0")/.."
mkdir -p logs/all checkpoint figures_out

# ---- common training config (formal, not smoke) ----------------------------
CFG="device=$DEVICE model.dim=192 model.depth=4 model.heads=6 \
train.batch_size=128 train.total_steps=$STEPS train.warmup_steps=$((STEPS/10)) \
train.eval_every=$((STEPS/8)) train.log_every=$((STEPS/10)) train.num_workers=4 \
paths.out_dir=logs/all"

# ---- experiment registry: "id|description|extra --set args" ----------------
# Core subset (ONLY=core runs just these three).
CORE=(
"2_1_transformer|[2.1] MAIN: dsRNATransformer (conv_stem + features)|model.arch=transformer model.conv_stem=true data.use_features=true"
"3_2_nofeat|[3.2] ablation: no thermo/tiling features|model.arch=transformer model.conv_stem=true data.use_features=false"
"4_1_genedisjoint|[4.1] robustness: gene-disjoint split|model.arch=transformer model.conv_stem=true data.use_features=true data.split=gene"
)
# Full extra experiments.
EXTRA=(
"2_2_cnnbilstm|[2.2] baseline: CNN-BiLSTM (dsRNAPredictor arch)|model.arch=cnn_bilstm model.dim=128"
"3_1_nostem|[3.1] ablation: no conv_stem (plain Transformer)|model.arch=transformer model.conv_stem=false data.use_features=true"
"3_3_kmer3|[3.3] ablation: 3-mer tokenization|model.arch=transformer model.conv_stem=true data.kmer=3"
"3_4_clspool|[3.4] ablation: CLS pooling|model.arch=transformer model.conv_stem=true data.use_features=true model.pool=cls"
"3_5a_dim96|[3.5] scale: dim=96 depth=2|model.arch=transformer model.conv_stem=true data.use_features=true model.dim=96 model.depth=2"
"3_5b_dim384|[3.5] scale: dim=384 depth=6|model.arch=transformer model.conv_stem=true data.use_features=true model.dim=384 model.depth=6"
"4_2a_t30|[4.2] threshold dali11>30%|model.arch=transformer model.conv_stem=true data.use_features=true data.binary_manifest=manifest_binary_t30.csv"
"4_2b_t50|[4.2] threshold dali11>50%|model.arch=transformer model.conv_stem=true data.use_features=true data.binary_manifest=manifest_binary_t50.csv"
"2_3_dsrnapredictor|[2.3] faithful dsRNAPredictor CNN-BiLSTM (Adam lr=1e-4, batch=32)|model.arch=cnn_faithful model.dim=100 train.optimizer=adam train.lr=0.0001 train.batch_size=32"
)

if [ "$ONLY" = "core" ]; then EXPS=("${CORE[@]}"); else EXPS=("${CORE[@]}" "${EXTRA[@]}"); fi
TOTAL=${#EXPS[@]}
START=$(date +%s)

echo "==============================================================================="
echo " dsRNATransformer — binary experiment suite"
echo " device=$DEVICE  steps=$STEPS  scope=$ONLY  experiments=$TOTAL"
echo " started $(date '+%Y-%m-%d %H:%M:%S')"
echo "==============================================================================="

# count how many experiments are already finished (resume support)
done_cnt=0
for e in "${EXPS[@]}"; do
  n="${e%%|*}"
  grep -qE "\[final\]" "logs/all/$n.log" 2>/dev/null && done_cnt=$((done_cnt+1))
done
echo " resume check: $done_cnt/$TOTAL already complete, $((TOTAL-done_cnt)) to run"

i=0
for e in "${EXPS[@]}"; do
  i=$((i+1))
  name="${e%%|*}"; rest="${e#*|}"; desc="${rest%%|*}"; sets="${rest#*|}"
  now=$(date +%s); el=$(( (now-START)/60 ))
  echo ""
  echo "-------------------------------------------------------------------------------"
  echo " [$i/$TOTAL] $desc"
  echo "   run_name=$name   (elapsed ${el} min)"
  # --- resume: skip experiments whose log already contains a [final] line ---
  if grep -qE "\[final\]" "logs/all/$name.log" 2>/dev/null; then
    fin=$(grep -E "\[final\]" "logs/all/$name.log" | tail -1)
    echo "   SKIP (already complete)"
    echo "   >>> RESULT: ${fin#*\[final\] }"
    continue
  fi
  echo "   config: $sets"
  echo "-------------------------------------------------------------------------------"
  # drop any stale partial log so completion detection stays accurate on re-run
  rm -f "logs/all/$name.log"
  $PY scripts/train_binary.py --config config/rnai.yaml --set $CFG $sets \
      run_name="$name" paths.ckpt_dir="checkpoint/$name" 2>&1 \
    | grep -vE "UserWarning|warnings.warn|enable_nested" \
    | sed "s/^/   [$name] /"
  fin=$(grep -E "\[final\]" "logs/all/$name.log" 2>/dev/null | tail -1)
  echo "   >>> RESULT: ${fin#*\[final\] }"
done

# ---- analysis / figures -----------------------------------------------------
echo ""
echo "==============================================================================="
echo " Generating figures + tables (Fig 1-7, Tables 2-5)"
echo "==============================================================================="
$PY scripts/plot_dataset.py   2>&1 | grep -vE "UserWarning|warnings.warn" | sed 's/^/   /'
$PY scripts/plot_binary.py    2>&1 | grep -vE "UserWarning|warnings.warn" | sed 's/^/   /'
$PY scripts/plot_interpret.py 2>&1 | grep -vE "UserWarning|warnings.warn" | sed 's/^/   /'

# ---- final summary table ----------------------------------------------------
echo ""
echo "==============================================================================="
echo " ALL DONE — summary of final metrics"
echo "==============================================================================="
printf "%-20s %-8s %-8s %-8s %-8s\n" "experiment" "Acc" "AUC" "MCC" "F1"
for e in "${EXPS[@]}"; do
  name="${e%%|*}"
  fin=$(grep -E "\[final\]" "logs/all/$name.log" 2>/dev/null | tail -1)
  acc=$(echo "$fin" | grep -oE '"Acc": [0-9.]+' | grep -oE '[0-9.]+')
  auc=$(echo "$fin" | grep -oE '"AUC": [0-9.]+' | grep -oE '[0-9.]+')
  mcc=$(echo "$fin" | grep -oE '"MCC": [0-9.]+' | grep -oE '[0-9.]+')
  f1=$(echo "$fin" | grep -oE '"F1": [0-9.]+' | grep -oE '[0-9.]+')
  printf "%-20s %-8s %-8s %-8s %-8s\n" "$name" "${acc:-NA}" "${auc:-NA}" "${mcc:-NA}" "${f1:-NA}"
done
el=$(( ($(date +%s)-START)/60 ))
echo "-------------------------------------------------------------------------------"
echo " total wall time: ${el} min   |   logs: logs/all/   |   figures: paper/figures/"
echo "==============================================================================="
