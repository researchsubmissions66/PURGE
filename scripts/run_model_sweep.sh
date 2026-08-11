#!/bin/bash

DATASETS=("PANDA" "BRACS")
MODELS=("abmil" "meanmil" "transmil")
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 15))
REM=$((SLURM_ARRAY_TASK_ID % 15))
MODEL_IDX=$((REM / 5))
FOLD=$((REM % 5))

DATASET=${DATASETS[$DATASET_IDX]}
MODEL_TYPE=${MODELS[$MODEL_IDX]}
ENCODER="features_hoptimus0"

echo "Running Model Sweep Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset: $DATASET"
echo "Model: $MODEL_TYPE"
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
LOCAL_ENCODER_DIR="/tmp/modelsweep_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$LOCAL_ENCODER_DIR"

echo "Extracting required filenames from metadata..."
awk -F, -v d="$DATASET" 'NR>1 && $9==d {print $1}' data/multi_benchmark_metadata.csv | sort | uniq > "$LOCAL_ENCODER_DIR/required_files.txt"
sed -i 's/\.[^.]*$//' "$LOCAL_ENCODER_DIR/required_files.txt"
sed -i 's/$/.h5/' "$LOCAL_ENCODER_DIR/required_files.txt"

echo "⚡ Caching $ENCODER features for $DATASET from Lustre to local NVMe SSD..."
cat "$LOCAL_ENCODER_DIR/required_files.txt" | xargs -I {} -P 16 cp -n "$ORIG_ENCODER_DIR/{}" "$LOCAL_ENCODER_DIR/" 2>/dev/null

# Evaluate Baseline
BASELINE_OUT="results/modelsweep_baseline_${DATASET}_${MODEL_TYPE}_${ENCODER}_fold${FOLD}.json"
if [ ! -f "$BASELINE_OUT" ]; then
    echo "Evaluating Baseline ($MODEL_TYPE) for $DATASET..."
    python scripts/train_mil.py \
        --encoder_dir "$LOCAL_ENCODER_DIR" \
        --dataset "$DATASET" \
        --model_type "$MODEL_TYPE" \
        --fold $FOLD \
        --epochs 15 \
        --patience 5 \
        --output_json "$BASELINE_OUT"
fi

# Evaluate Organ Erasures
ORGANS=("PROSTATE" "BREAST")
for ORGAN in "${ORGANS[@]}"; do
    UNLEARNED_OUT="results/modelsweep_${ORGAN}_on_${DATASET}_${MODEL_TYPE}_${ENCODER}_fold${FOLD}.json"
    if [ ! -f "$UNLEARNED_OUT" ]; then
        echo "Evaluating Unlearned Features ($ORGAN Removed, $MODEL_TYPE) for $DATASET..."
        python scripts/train_mil.py \
            --encoder_dir "$LOCAL_ENCODER_DIR" \
            --dataset "$DATASET" \
            --model_type "$MODEL_TYPE" \
            --unlearner "results/unlearners/${ENCODER}_${ORGAN}_k500.pt" \
            --k 50 \
            --fold $FOLD \
            --epochs 15 \
            --patience 5 \
            --output_json "$UNLEARNED_OUT"
    fi
done

echo "Cleaning up local cache to save space..."
rm -rf "$LOCAL_ENCODER_DIR"
echo "Job finished successfully."
