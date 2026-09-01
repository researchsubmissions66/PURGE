#!/bin/bash
# Shared configuration for every SLURM launcher. Source this; do not run it.
#
# get_encoder_dir() lives here in one place because the patch-size rules differ
# per foundation model and per cohort, and a launcher that hardcodes 224px
# silently points at a directory that does not exist for the 256/512px models.

DATASETS=("PANDA" "BACH" "BRACS" "UBC-OCEAN" "TCGA-LUNG" "TCGA-BRCA")
FMS=("features_hoptimus0" "features_virchow" "features_virchow2" "features_gpfm" \
     "features_uni_v2" "features_gigapath" "features_ctranspath" "features_conch_v15")

BASE_DIR="${PURGE_BASE_DIR:-/work/hdd/bhwm}"
OUTPUT_DIR="${PURGE_OUTPUT_DIR:-results/sweep}"
METADATA="${PURGE_METADATA:-data/multi_benchmark_metadata.csv}"

SLURM_ACCOUNT="${PURGE_SLURM_ACCOUNT:-bhwm-delta-gpu}"
# Comma-separated list: Slurm schedules on whichever partition frees first.
# A40 (48GB) is usually far less contended than A100 and this workload is
# I/O-bound on h5 reads with batch-size-1 MIL, so the throughput loss is small.
SLURM_PARTITION="${PURGE_SLURM_PARTITION:-gpuA40x4,gpuA100x4}"
SLURM_MEM="${PURGE_SLURM_MEM:-64G}"
SLURM_CPUS="${PURGE_SLURM_CPUS:-8}"
MODULE_CMD="${PURGE_MODULE_CMD:-module load pytorch-conda}"

# Models extracted at 256px; conch_v15 uses 512px on TCGA.
_is_256px() {
    case "$1" in
        features_uni_v2|features_gigapath|features_ctranspath|features_conch_v15) return 0 ;;
        *) return 1 ;;
    esac
}

get_encoder_dir() {
    local dataset=$1 fm=$2 root
    if [[ "$dataset" == TCGA* ]]; then
        root="${BASE_DIR}/trident_features/master_benchmark"
    else
        root="${BASE_DIR}/${dataset}"
    fi

    if [[ "$fm" == "features_conch_v15" && "$dataset" == TCGA* ]]; then
        echo "${root}/20x_512px_0px_overlap/${fm}"
    elif _is_256px "$fm"; then
        echo "${root}/20x_256px_0px_overlap/${fm}"
    else
        echo "${root}/20x_224px_0px_overlap/${fm}"
    fi
}

# submit <job_name> <time_limit> <command...>
submit() {
    local job_name=$1 time_limit=$2
    shift 2
    mkdir -p logs
    sbatch --job-name="$job_name" \
           --output="logs/${job_name}_%j.log" \
           --account="$SLURM_ACCOUNT" \
           --partition="$SLURM_PARTITION" \
           --mem="$SLURM_MEM" \
           --cpus-per-task="$SLURM_CPUS" \
           --gpus=1 \
           --nodes=1 \
           --time="$time_limit" \
           --wrap="$MODULE_CMD && $*"
}
