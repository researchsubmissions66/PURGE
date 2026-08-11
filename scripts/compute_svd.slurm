#!/bin/bash
#SBATCH --job-name=svd_precompute
#SBATCH --output=logs/svd_%A_%a.out
#SBATCH --error=logs/svd_%A_%a.err
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --account=bhwm-delta-gpu

module load pytorch-conda
conda activate base

ORGANS=("PROSTATE" "BREAST" "LUNG" "OVARIAN")
ENCODERS=("features_hoptimus0" "features_gpfm" "features_virchow2")

ENCODER_IDX=$((SLURM_ARRAY_TASK_ID / 4))
ORGAN_IDX=$((SLURM_ARRAY_TASK_ID % 4))

ENCODER=${ENCODERS[$ENCODER_IDX]}
ORGAN=${ORGANS[$ORGAN_IDX]}

echo "Running SVD for $ORGAN on $ENCODER..."
python scripts/compute_all_unlearners.py --encoder "$ENCODER" --organ "$ORGAN"
