#!/bin/bash

# Sweep evaluating Naive Noise Perturbations vs Baseline
# Array index logic:
# 0-19: Gaussian Noise (sigma=0.5, 1.0)
# 20-39: Dropout (p=0.5, 0.9)
# Inside each block: 2 datasets (PANDA, BRACS) x 5 folds = 10 tasks.
# 4 states * 10 tasks = 40 tasks (SLURM_ARRAY_TASK_ID 0-39)

DATASETS=("PANDA" "BRACS")
ENCODER="features_hoptimus0"

TASK_ID=$SLURM_ARRAY_TASK_ID
FOLD=$((TASK_ID % 5))
DATASET_IDX=$(((TASK_ID / 5) % 2))
DATASET=${DATASETS[$DATASET_IDX]}

STATE_IDX=$((TASK_ID / 10))

if [ $STATE_IDX -eq 0 ]; then
    METHOD="gaussian"
    SIGMA="0.5"
elif [ $STATE_IDX -eq 1 ]; then
    METHOD="gaussian"
    SIGMA="1.0"
elif [ $STATE_IDX -eq 2 ]; then
    METHOD="dropout"
    P="0.5"
elif [ $STATE_IDX -eq 3 ]; then
    METHOD="dropout"
    P="0.9"
fi

echo "Running Noise Sweep Task ID: $TASK_ID"
echo "Dataset: $DATASET"
echo "Method: $METHOD"
if [ "$METHOD" == "gaussian" ]; then echo "Sigma: $SIGMA"; else echo "P: $P"; fi
echo "Fold: $FOLD"

if [ "$DATASET" == "PANDA" ]; then
    ROOT_DIR="/work/hdd/bhwm/PANDA/20x_224px_0px_overlap"
elif [ "$DATASET" == "BRACS" ]; then
    ROOT_DIR="/work/hdd/bhwm/BRACS/20x_224px_0px_overlap"
fi

ORIG_ENCODER_DIR="${ROOT_DIR}/${ENCODER}"
LOCAL_ENCODER_DIR="/tmp/noisesweep_${SLURM_JOB_ID}_${TASK_ID}"

mkdir -p "$LOCAL_ENCODER_DIR"

echo "Extracting required filenames from metadata..."
awk -F, -v d="$DATASET" 'NR>1 && $9==d {print $1}' data/multi_benchmark_metadata.csv | sort | uniq > "$LOCAL_ENCODER_DIR/required_files.txt"
sed -i 's/\.[^.]*$//' "$LOCAL_ENCODER_DIR/required_files.txt"
sed -i 's/$/.h5/' "$LOCAL_ENCODER_DIR/required_files.txt"

echo "⚡ Caching $ENCODER features for $DATASET from Lustre to local NVMe SSD..."
cat "$LOCAL_ENCODER_DIR/required_files.txt" | xargs -I {} -P 16 cp -n "$ORIG_ENCODER_DIR/{}" "$LOCAL_ENCODER_DIR/" 2>/dev/null

if [ "$METHOD" == "gaussian" ]; then
    UNLEARNED_OUT="results/noisesweep_gaussian_s${SIGMA}_on_${DATASET}_abmil_fold${FOLD}.json"
    EXTRA_ARGS="--unlearn_method gaussian --sigma $SIGMA"
else
    UNLEARNED_OUT="results/noisesweep_dropout_p${P}_on_${DATASET}_abmil_fold${FOLD}.json"
    EXTRA_ARGS="--unlearn_method dropout --dropout_p $P"
fi

if [ ! -f "$UNLEARNED_OUT" ]; then
    echo "Evaluating Naive Baseline..."
    # Notice we pass --unlearner "dummy" because the code expects the arg, but doesn't load a file for noise
    python scripts/train_mil.py \
        --encoder_dir "$LOCAL_ENCODER_DIR" \
        --dataset "$DATASET" \
        --model_type "abmil" \
        --unlearner "dummy" \
        $EXTRA_ARGS \
        --fold $FOLD \
        --epochs 15 \
        --patience 5 \
        --output_json "$UNLEARNED_OUT"
fi

echo "Cleaning up local cache to save space..."
rm -rf "$LOCAL_ENCODER_DIR"
echo "Job finished successfully."
