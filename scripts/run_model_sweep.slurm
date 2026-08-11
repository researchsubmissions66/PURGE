#!/bin/bash
#SBATCH --job-name=model_sweep
#SBATCH --output=logs/modelsweep_%A_%a.out
#SBATCH --error=logs/modelsweep_%A_%a.err
#SBATCH --array=0-29
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --account=bhwm-delta-gpu

module load pytorch-conda
conda activate base

bash scripts/run_model_sweep.sh
