#!/bin/bash

# Sweep evaluating Wiener Filter vs Baseline
DATASETS=("PANDA" "BRACS")
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 5))
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
DATASET=${DATASETS[$DATASET_IDX]}

ENCODER="features_hoptimus0"

echo "Running Wiener Sweep Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset: $DATASET"
echo "Encoder: $ENCODER"
echo "Fold: $FOLD"

if [ "$DATASET" == "PANDA" ]; then
    ROOT_DIR="/work/hdd/bhwm/PANDA/20x_224px_0px_overlap"
    TARGET_ORGAN="PROSTATE"
elif [ "$DATASET" == "BRACS" ]; then
    ROOT_DIR="/work/hdd/bhwm/BRACS/20x_224px_0px_overlap"
    TARGET_ORGAN="BREAST"
fi

ORIG_ENCODER_DIR="${ROOT_DIR}/${ENCODER}"
LOCAL_ENCODER_DIR="/tmp/wienersweep_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$LOCAL_ENCODER_DIR"

echo "Extracting required filenames from metadata..."
awk -F, -v d="$DATASET" 'NR>1 && $9==d {print $1}' data/multi_benchmark_metadata.csv | sort | uniq > "$LOCAL_ENCODER_DIR/required_files.txt"
sed -i 's/\.[^.]*$//' "$LOCAL_ENCODER_DIR/required_files.txt"
sed -i 's/$/.h5/' "$LOCAL_ENCODER_DIR/required_files.txt"

echo "⚡ Caching $ENCODER features for $DATASET from Lustre to local NVMe SSD..."
cat "$LOCAL_ENCODER_DIR/required_files.txt" | xargs -I {} -P 16 cp -n "$ORIG_ENCODER_DIR/{}" "$LOCAL_ENCODER_DIR/" 2>/dev/null

# Evaluate Organ Erasure using Wiener Filter
ORGANS=("PROSTATE" "BREAST")
for ORGAN in "${ORGANS[@]}"; do
    UNLEARNED_OUT="results/wienersweep_${ORGAN}_on_${DATASET}_abmil_${ENCODER}_fold${FOLD}.json"
    if [ ! -f "$UNLEARNED_OUT" ]; then
        echo "Evaluating Wiener Filter Unlearned Features ($ORGAN Removed) for $DATASET..."
        python scripts/train_mil.py \
            --encoder_dir "$LOCAL_ENCODER_DIR" \
            --dataset "$DATASET" \
            --model_type "abmil" \
            --unlearner "results/unlearners/wiener_${ENCODER}_${ORGAN}_k500.pt" \
            --unlearn_method "wiener" \
            --tau 0.1 \
            --fold $FOLD \
            --epochs 15 \
            --patience 5 \
            --output_json "$UNLEARNED_OUT"
    fi
done

echo "Cleaning up local cache to save space..."
rm -rf "$LOCAL_ENCODER_DIR"
echo "Job finished successfully."
