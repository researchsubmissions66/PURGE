#!/bin/bash
#SBATCH --job-name=compute_wiener
#SBATCH --output=logs/compute_wiener_%j.out
#SBATCH --error=logs/compute_wiener_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --account=bhwm-delta-gpu

module load pytorch-conda
conda activate base

python scripts/compute_wiener_unlearners.py --encoder "features_hoptimus0" --k_components 500
