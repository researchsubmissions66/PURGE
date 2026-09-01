#!/bin/bash
# Submit the fast pooled-feature validation to a GPU node.
#
# Runs on a compute node rather than the login node: feature loading is
# I/O-heavy on /work/hdd and the login node enforces a 30-minute CPU limit.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

# The interactive partitions are near-empty (PD~1) while the batch GPU queues
# are thousands deep. 1h cap, but feature caching is incremental per dataset,
# so a timeout just means resubmitting and picking up the cached ones.
PARTITION="${PURGE_QUICK_PARTITION:-gpuA40x4-interactive,gpuA100x4-interactive}"
WALLTIME="${WALLTIME:-01:00:00}"

TARGET="${TARGET:-TCGA-LUNG}"
CONTROLS="${CONTROLS:---control UBC-OCEAN --control BACH}"
RANKS="${RANKS:-16 64 256}"
STEPS="${STEPS:-300}"
MAX_SLIDES="${MAX_SLIDES:-600}"
PATCHES="${PATCHES:-256}"
WORKERS="${WORKERS:-48}"

mkdir -p logs results/quick

sbatch --job-name="quickval_${TARGET}" \
       --output="logs/quickval_${TARGET}_%j.log" \
       --account="$SLURM_ACCOUNT" \
       --partition="$PARTITION" \
       --mem=96G \
       --cpus-per-task=16 \
       --gpus=1 \
       --nodes=1 \
       --time="$WALLTIME" \
       --wrap="$MODULE_CMD && python -u scripts/quick_validate.py \
                --target $TARGET $CONTROLS \
                --ranks $RANKS --steps $STEPS \
                --max_slides $MAX_SLIDES --patches $PATCHES --workers $WORKERS \
                --out results/quick/quick_validate_${TARGET}.json ${EXTRA:-}"
