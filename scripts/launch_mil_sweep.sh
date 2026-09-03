#!/bin/bash
# MIL confirmation sweep: the pooled-feature sweep is a PROXY, this is the real thing.
#
# Why both: erasure is linear and mean pooling is linear, so they commute -
# a probe on erased slide-means IS MeanMIL on erased patches. That does NOT hold
# for ABMIL or TransMIL, whose pooling is a nonlinear function of the patches and
# is re-learned after erasure. Headline claims need these.
#
# Uses ALL slides (train_mil.py reads the full bag for every slide in the split).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

MODELS=("${MIL_MODELS[@]:-abmil transmil meanmil}")
DATASETS_MIL=("${MIL_DATASETS[@]:-TCGA-LUNG PANDA BRACS}")
ORGANS=("${MIL_ORGANS[@]:-LUNG PROSTATE BREAST}")
FM="${FM:-features_virchow2}"
K="${K:-64}"
FOLDS="${FOLDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-15}"
OUT="${OUT:-results/mil_sweep}"
UDIR="${UDIR:-results/unlearners}"

mkdir -p logs "$OUT" "$UDIR"

# ---- 1. fit one eraser per organ (all slides, patch-level) --------------- #
for organ in "${ORGANS[@]}"; do
    out="$UDIR/${FM}_${organ}_k${K}.pt"
    [[ -f "$out" ]] && continue
    enc=$(get_encoder_dir PANDA "$FM")
    submit "fit_${organ}" "04:00:00" \
        "python scripts/fit_unlearner.py --forget_organ $organ \
         --encoder_dir $enc --metadata $METADATA --method svd --k $K \
         --max_slides 1000 --patches_per_slide 64 --output $out"
done

# ---- 2. baseline + erased, every model x dataset x fold ----------------- #
for fold in $FOLDS; do
  for model in "${MODELS[@]}"; do
    for ds in "${DATASETS_MIL[@]}"; do
      enc=$(get_encoder_dir "$ds" "$FM")
      [[ -d "$enc" ]] || continue

      base_json="$OUT/base_${ds}_${model}_f${fold}.json"
      if [[ ! -f "$base_json" ]]; then
        submit "mil_base_${ds}_${model}_f${fold}" "08:00:00" \
          "python scripts/train_mil.py --dataset $ds --encoder_dir $enc \
           --metadata $METADATA --fold $fold --model_type $model \
           --epochs $EPOCHS --output_json $base_json"
      fi

      for organ in "${ORGANS[@]}"; do
        er="$UDIR/${FM}_${organ}_k${K}.pt"
        out_json="$OUT/erase-${organ}_${ds}_${model}_f${fold}.json"
        [[ -f "$out_json" ]] && continue
        submit "mil_${organ}_${ds}_${model}_f${fold}" "08:00:00" \
          "python scripts/train_mil.py --dataset $ds --encoder_dir $enc \
           --metadata $METADATA --fold $fold --model_type $model \
           --epochs $EPOCHS --unlearn_method svd --unlearner $er --k $K \
           --output_json $out_json"
      done
    done
  done
done
echo "MIL sweep submitted. Collect with: python scripts/collect_mil.py"
