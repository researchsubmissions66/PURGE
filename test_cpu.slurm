#!/bin/bash
#SBATCH --job-name=test_cpu
#SBATCH --partition=gpuA100x4
#SBATCH --account=bhwm-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=test_cpu.out

echo "Base:"
python -c "import os; print('os.cpu_count() =', os.cpu_count())"

echo "srun default:"
srun python -c "import os; print('os.cpu_count() =', os.cpu_count())" || echo "srun default failed"

echo "srun --exact:"
srun --exact python -c "import os; print('os.cpu_count() =', os.cpu_count())" || echo "srun exact failed"

echo "srun --overlap:"
srun --overlap python -c "import os; print('os.cpu_count() =', os.cpu_count())" || echo "srun overlap failed"
