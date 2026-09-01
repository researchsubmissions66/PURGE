#!/bin/bash
# Cross-evaluation: for each fitted eraser, retrain a probe FROM SCRATCH on every
# dataset through it. Only these numbers are evidence of erasure.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

NUM_FOLDS="${NUM_FOLDS:-5}"
FMS=("${EVAL_FMS[@]:-features_virchow2}")
ORGANS=("PROSTATE" "BREAST" "LUNG" "OVARIAN")
K="${K:-64}"
EPOCHS="${EPOCHS:-15}"

echo "Launching cross-evaluations..."
for fold in $(seq 0 $((NUM_FOLDS - 1))); do
    for fm in "${FMS[@]}"; do
        for organ in "${ORGANS[@]}"; do
            eraser="${OUTPUT_DIR}/${fm}/unlearners/${organ}_k${K}.pt"
            [[ -f "$eraser" ]] || { echo "  skip $organ: not fitted yet"; continue; }

            for dataset in "${DATASETS[@]}"; do
                enc_dir=$(get_encoder_dir "$dataset" "$fm")
                [[ -d "$enc_dir" ]] || continue
                out_json="${OUTPUT_DIR}/${fm}/eval_target-${organ}_control-${dataset}_fold${fold}.json"
                [[ -f "$out_json" ]] && continue

                submit "eval_T-${organ}_C-${dataset}_${fm}_f${fold}" "06:00:00" \
                    "python scripts/train_mil.py --dataset $dataset --encoder_dir $enc_dir \
                     --metadata $METADATA --fold $fold \
                     --unlearn_method svd --unlearner $eraser --k $K \
                     --epochs $EPOCHS --output_json $out_json"
            done
        done
    done
done
