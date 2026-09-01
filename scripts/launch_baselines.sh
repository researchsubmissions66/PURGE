#!/bin/bash
# Baseline MIL probes with no erasure. Establishes the P_T(E) reference that
# every degradation number is measured against.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

NUM_FOLDS="${NUM_FOLDS:-5}"
FMS=("${BASELINE_FMS[@]:-${FMS[@]}}")

echo "Launching baseline probes over ${NUM_FOLDS} fold(s)..."
for fold in $(seq 0 $((NUM_FOLDS - 1))); do
    for fm in "${FMS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            enc_dir=$(get_encoder_dir "$dataset" "$fm")
            [[ -d "$enc_dir" ]] || { echo "  skip $dataset/$fm: no $enc_dir"; continue; }

            out_json="${OUTPUT_DIR}/${fm}/baseline_${dataset}_fold${fold}.json"
            model_pt="${OUTPUT_DIR}/${fm}/models/base_${dataset}_fold${fold}.pt"
            [[ -f "$out_json" ]] && continue

            submit "base_${dataset}_${fm}_f${fold}" "04:00:00" \
                "python scripts/train_mil.py --dataset $dataset --encoder_dir $enc_dir \
                 --metadata $METADATA --fold $fold --output_json $out_json \
                 --save_model_path $model_pt"
        done
    done
done
