#!/bin/bash
#SBATCH --job-name=svd_k500
#SBATCH --output=logs/svd_k500_%A_%a.out
#SBATCH --error=logs/svd_k500_%A_%a.err
#SBATCH --array=0-3
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
ORGAN=${ORGANS[$SLURM_ARRAY_TASK_ID]}
ENCODER="features_hoptimus0"

echo "Running SVD (k=500) for $ORGAN on $ENCODER..."
python scripts/compute_all_unlearners.py --encoder "$ENCODER" --organ "$ORGAN" --k_components 500
