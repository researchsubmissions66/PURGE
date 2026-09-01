#!/bin/bash
# End-to-end smoke test on a single dataset pair. Run interactively on a GPU node
# before submitting a sweep - it exercises the same code paths in a few minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

FM="${FM:-features_virchow2}"
TARGET="${TARGET:-BACH}"
CONTROL="${CONTROL:-TCGA-LUNG}"
K="${K:-64}"
ORGAN="${ORGAN:-LUNG}"
OUT="${OUT:-results/smoke}"

source scripts/common.sh
TARGET_DIR=$(get_encoder_dir "$TARGET" "$FM")
CONTROL_DIR=$(get_encoder_dir "$CONTROL" "$FM")

echo "== unit tests =="
python -m pytest tests/ -q

echo
echo "== fitting affine-SVD eraser: organ=$ORGAN k=$K =="
python scripts/fit_unlearner.py \
    --forget_organ "$ORGAN" --encoder_dir "$TARGET_DIR" \
    --metadata data/multi_benchmark_metadata.csv \
    --method svd --k "$K" --output "${OUT}/unlearners/${ORGAN}_k${K}.pt"

ERASER="${OUT}/unlearners/${ORGAN}_k${K}.pt"

echo
echo "== retraining a fresh probe on the control through the eraser =="
python scripts/train_mil.py \
    --dataset "$CONTROL" --encoder_dir "$CONTROL_DIR" \
    --metadata data/multi_benchmark_metadata.csv \
    --fold 0 --epochs 2 \
    --unlearn_method svd --unlearner "$ERASER" --k "$K" \
    --output_json "${OUT}/eval_${ORGAN}_control-${CONTROL}_fold0.json"

echo
echo "Smoke test finished. Results under ${OUT}/"
