#!/bin/bash

# Ensure PYTHONPATH is set so modules resolve correctly
export PYTHONPATH=$(pwd)

# Select encoders to test for the pilot
ENCODERS=("features_hoptimus0" "features_virchow2" "features_gpfm")
BASE_DIR="/work/hdd/bhwm/trident_features/master_benchmark/20x_224px_0px_overlap"

# Define Local NVMe cache directory using the SLURM Job ID (or a generic fallback)
CACHE_DIR="/tmp/purge_cache_${SLURM_JOB_ID:-manual}"
mkdir -p "$CACHE_DIR"

echo "Extracting required filenames from metadata..."
# Grab just the filename column, skip header, and get unique names
awk -F, 'NR>1 {print $3}' data/metadata.csv | sort | uniq > "$CACHE_DIR/required_files.txt"
NUM_FILES=$(wc -l < "$CACHE_DIR/required_files.txt")
echo "Found $NUM_FILES unique slides required for this run."

for ENCODER in "${ENCODERS[@]}"
do
    echo "======================================"
    echo "Running Pilot for Encoder: $ENCODER"
    echo "======================================"
    
    ORIG_ENCODER_DIR="${BASE_DIR}/${ENCODER}"
    LOCAL_ENCODER_DIR="${CACHE_DIR}/${ENCODER}"
    mkdir -p "$LOCAL_ENCODER_DIR"
    
    echo "⚡ Caching $ENCODER features from Lustre to local NVMe SSD..."
    # Read the required filenames and copy them using 16 parallel threads. 
    # Errors (e.g. missing files) are sent to dev/null to prevent log spam.
    cat "$CACHE_DIR/required_files.txt" | xargs -I {} -P 16 cp -n "$ORIG_ENCODER_DIR/{}" "$LOCAL_ENCODER_DIR/" 2>/dev/null
    
    # 1. Fit unlearner for Kidney
    echo "Fitting unlearner for Kidney..."
    python scripts/fit_unlearner.py \
        --encoder_dir "$LOCAL_ENCODER_DIR" \
        --forget_organ "KIDNEY" \
        --output "results/unlearners/${ENCODER}_kidney.pt"
        
    # 2. Evaluate Baseline ABMIL on Lung
    echo "Evaluating Baseline (Original Features) for Lung LUAD/LUSC..."
    python scripts/train_abmil.py \
        --encoder_dir "$LOCAL_ENCODER_DIR" \
        --fold 0
        
    # 3. Evaluate Unlearned ABMIL on Lung
    echo "Evaluating Unlearned Features (Kidney Removed) for Lung LUAD/LUSC..."
    python scripts/train_abmil.py \
        --encoder_dir "$LOCAL_ENCODER_DIR" \
        --unlearner "results/unlearners/${ENCODER}_kidney.pt" \
        --fold 0

    echo "Cleaning up local cache for $ENCODER to save space..."
    rm -rf "$LOCAL_ENCODER_DIR"
done

echo "Pilot execution complete."

