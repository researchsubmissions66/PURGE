#!/bin/bash
#SBATCH --job-name=unlearn_pilot
#SBATCH --account=bhwm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=32g
#SBATCH --output=unlearn_pilot_%j.out

echo "Job starting on $(hostname)"

# Load the recommended PyTorch conda module for NCSA Delta
module load pytorch-conda

cd /u/dchanda/PURGE

bash scripts/run_pilot.sh

echo "Job finished successfully."
