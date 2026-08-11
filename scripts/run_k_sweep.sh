#!/bin/bash

# Datasets to evaluate for the K-sweep
DATASETS=("PANDA" "BRACS" "TCGA")
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 5))
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
DATASET=${DATASETS[$DATASET_IDX]}

ENCODER="features_hoptimus0"

echo "Running K-Sweep Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset: $DATASET"
echo "Encoder: $ENCODER"
echo "Fold: $FOLD"

# Dynamic routing
if [ "$DATASET" == "PANDA" ]; then
    ROOT_DIR="/work/hdd/bhwm/PANDA/20x_224px_0px_overlap"
    TARGET_ORGAN="PROSTATE"
elif [ "$DATASET" == "BRACS" ]; then
    ROOT_DIR="/work/hdd/bhwm/BRACS/20x_224px_0px_overlap"
    TARGET_ORGAN="BREAST"
elif [ "$DATASET" == "TCGA" ]; then
    ROOT_DIR="/work/hdd/bhwm/trident_features/master_benchmark/20x_224px_0px_overlap"
    TARGET_ORGAN="LUNG" 
fi

ORIG_ENCODER_DIR="${ROOT_DIR}/${ENCODER}"
LOCAL_ENCODER_DIR="/tmp/ksweep_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$LOCAL_ENCODER_DIR"

echo "Extracting required filenames from metadata..."
awk -F, -v d="$DATASET" 'NR>1 && $9==d {print $1}' data/multi_benchmark_metadata.csv | sort | uniq > "$LOCAL_ENCODER_DIR/required_files.txt"
sed -i 's/\.[^.]*$//' "$LOCAL_ENCODER_DIR/required_files.txt"
sed -i 's/$/.h5/' "$LOCAL_ENCODER_DIR/required_files.txt"

echo "⚡ Caching $ENCODER features for $DATASET from Lustre to local NVMe SSD..."
cat "$LOCAL_ENCODER_DIR/required_files.txt" | xargs -I {} -P 16 cp -n "$ORIG_ENCODER_DIR/{}" "$LOCAL_ENCODER_DIR/" 2>/dev/null

K_VALUES=(1 5 10 50 100 200 500)

for K in "${K_VALUES[@]}"; do
    UNLEARNED_OUT="results/ksweep_${TARGET_ORGAN}_on_${DATASET}_${ENCODER}_k${K}_fold${FOLD}.json"
    if [ ! -f "$UNLEARNED_OUT" ]; then
        echo "Evaluating Unlearned Features (k=$K $TARGET_ORGAN Removed) for $DATASET..."
        python scripts/train_abmil.py \
            --encoder_dir "$LOCAL_ENCODER_DIR" \
            --dataset "$DATASET" \
            --unlearner "results/unlearners/${ENCODER}_${TARGET_ORGAN}_k500.pt" \
            --k $K \
            --fold $FOLD \
            --epochs 15 \
            --patience 5 \
            --output_json "$UNLEARNED_OUT"
    else
        echo "K-Sweep results for k=$K on $DATASET fold $FOLD already exist. Skipping..."
    fi
done

echo "Cleaning up local cache to save space..."
rm -rf "$LOCAL_ENCODER_DIR"
echo "Job finished successfully."
