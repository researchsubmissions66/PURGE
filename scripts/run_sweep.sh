#!/bin/bash

DATASETS=("PANDA" "BRACS" "TCGA" "UBC-OCEAN" "BACH")
ENCODERS=("features_hoptimus0" "features_gpfm" "features_virchow2")

# Math to map SLURM_ARRAY_TASK_ID (0-74)
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 15))
REM=$((SLURM_ARRAY_TASK_ID % 15))
ENCODER_IDX=$((REM / 5))
FOLD=$((REM % 5))

DATASET=${DATASETS[$DATASET_IDX]}
ENCODER=${ENCODERS[$ENCODER_IDX]}

echo "Running Sweep Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset: $DATASET"
echo "Encoder: $ENCODER"
echo "Fold: $FOLD"

# Dynamic routing
if [ "$DATASET" == "PANDA" ]; then
    ROOT_DIR="/work/hdd/bhwm/PANDA/20x_224px_0px_overlap"
    FORGET_ORGAN="PROSTATE"
elif [ "$DATASET" == "BRACS" ]; then
    ROOT_DIR="/work/hdd/bhwm/BRACS/20x_224px_0px_overlap"
    FORGET_ORGAN="BREAST"
elif [ "$DATASET" == "TCGA" ]; then
    ROOT_DIR="/work/hdd/bhwm/trident_features/master_benchmark/20x_224px_0px_overlap"
    FORGET_ORGAN="LUNG" 
elif [ "$DATASET" == "UBC-OCEAN" ]; then
    ROOT_DIR="/work/hdd/bhwm/UBC-OCEAN/20x_224px_0px_overlap"
    FORGET_ORGAN="OVARIAN"
elif [ "$DATASET" == "BACH" ]; then
    ROOT_DIR="/work/hdd/bhwm/BACH/20x_224px_0px_overlap"
    FORGET_ORGAN="BREAST"
fi

ORIG_ENCODER_DIR="${ROOT_DIR}/${ENCODER}"
LOCAL_ENCODER_DIR="/tmp/purge_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$LOCAL_ENCODER_DIR"

echo "Extracting required filenames from metadata..."
# Extract the filenames for this specific dataset
awk -F, -v d="$DATASET" 'NR>1 && $9==d {print $1}' data/multi_benchmark_metadata.csv | sort | uniq > "$LOCAL_ENCODER_DIR/required_files.txt"

# For external datasets, the filename in metadata is usually .tiff or .png, but features are .h5
# So we just extract the stem and append .h5
sed -i 's/\.[^.]*$//' "$LOCAL_ENCODER_DIR/required_files.txt"
sed -i 's/$/.h5/' "$LOCAL_ENCODER_DIR/required_files.txt"

echo "⚡ Caching $ENCODER features for $DATASET from Lustre to local NVMe SSD..."
cat "$LOCAL_ENCODER_DIR/required_files.txt" | xargs -I {} -P 16 cp -n "$ORIG_ENCODER_DIR/{}" "$LOCAL_ENCODER_DIR/" 2>/dev/null

# Unlearners are now pre-computed using SVD via compute_all_unlearners.py

BASELINE_OUT="results/baseline_${DATASET}_${ENCODER}_fold${FOLD}.json"
if [ ! -f "$BASELINE_OUT" ]; then
    echo "Evaluating Baseline (Original Features) for $DATASET..."
    python scripts/train_abmil.py \
        --encoder_dir "$LOCAL_ENCODER_DIR" \
        --dataset "$DATASET" \
        --fold $FOLD \
        --epochs 15 \
        --patience 5 \
        --output_json "$BASELINE_OUT"
else
    echo "Baseline results for $DATASET fold $FOLD already exist. Skipping..."
fi

ORGANS=("PROSTATE" "BREAST" "LUNG" "OVARIAN")

for TARGET_ORGAN in "${ORGANS[@]}"; do
    UNLEARNED_OUT="results/unlearned_${TARGET_ORGAN}_on_${DATASET}_${ENCODER}_fold${FOLD}.json"
    if [ ! -f "$UNLEARNED_OUT" ]; then
        echo "Evaluating Unlearned Features ($TARGET_ORGAN Removed) for $DATASET..."
        python scripts/train_abmil.py \
            --encoder_dir "$LOCAL_ENCODER_DIR" \
            --dataset "$DATASET" \
            --unlearner "results/unlearners/${ENCODER}_${TARGET_ORGAN}.pt" \
            --fold $FOLD \
            --epochs 15 \
            --patience 5 \
            --output_json "$UNLEARNED_OUT"
    else
        echo "Unlearned results for $TARGET_ORGAN on $DATASET fold $FOLD already exist. Skipping..."
    fi
done

echo "Cleaning up local cache to save space..."
rm -rf "$LOCAL_ENCODER_DIR"

echo "Job finished successfully."
