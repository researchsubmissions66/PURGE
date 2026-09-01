#!/bin/bash
# Fit one affine-SVD eraser per target organ.
#
# The adversarial eraser this script used to launch is gone: it did not converge,
# and initialising it from the SVD subspace made it walk away from the answer.
# See AGENTS.md. The closed-form fit below is what works.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

ORGANS=("PROSTATE" "BREAST" "LUNG" "OVARIAN")
FMS=("${ERASER_FMS[@]:-features_virchow2}")
K="${K:-64}"

echo "Fitting affine-SVD erasers (k=${K})..."
for fm in "${FMS[@]}"; do
    for organ in "${ORGANS[@]}"; do
        enc_dir=$(get_encoder_dir PANDA "$fm")   # any cohort dir: metadata carries the rest
        out="${OUTPUT_DIR}/${fm}/unlearners/${organ}_k${K}.pt"
        [[ -f "$out" ]] && continue

        submit "fit_${organ}_${fm}" "02:00:00" \
            "python scripts/fit_unlearner.py --forget_organ $organ \
             --encoder_dir $enc_dir --metadata $METADATA \
             --method svd --k $K --output $out"
    done
done
