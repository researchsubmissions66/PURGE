#!/bin/bash
# Launch the config-driven sweep.
#
# Structure follows the measured cost: with a warm cache a config takes ~9s, so
# all 31 runs are ~5 min of compute. The entire expense is warming feature caches
# for encoders not yet seen (~10 min each). So:
#
#   1. one cache-warm job per NEW encoder, throttled to the QOS limit
#   2. ONE run job that loops over every pending config
#
# No SLURM array: the interactive partitions cap SUBMITTED jobs per user at 2, and
# a 31-task array trips it. Resumable - run_config.py skips completed results, so
# re-running this after a timeout picks up exactly what is missing.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG_DIR="${CONFIG_DIR:-configs/sweeps}"
OUT_DIR="${OUT_DIR:-results/sweep_v2}"
CACHE_DIR="${CACHE_DIR:-results/quick/cache}"
ACCOUNT="${PURGE_SLURM_ACCOUNT:-bhwm-delta-gpu}"
PARTITION="${PURGE_SWEEP_PARTITION:-gpuA40x4-interactive,gpuA100x4-interactive}"
# Interactive only: the batch GPU queues are thousands deep, so a job there
# never starts. 50 min leaves margin inside the 1h interactive cap.
WALLTIME="${WALLTIME:-00:50:00}"
MAXJOBS="${MAXJOBS:-2}"
MODULE_CMD="${PURGE_MODULE_CMD:-module load pytorch-conda}"

mkdir -p logs "$OUT_DIR" "$CACHE_DIR"

wait_for_slot() {
    until [ "$(squeue -h -u "$USER" -o '%j' | grep -cE '^(warm_|purge_sweep)')" -lt "$MAXJOBS" ]; do
        sleep 30
    done
}

# ---- pending configs ---------------------------------------------------- #
PENDING=()
for c in "$CONFIG_DIR"/*.json; do
    rid=$(basename "$c" .json)
    r="$OUT_DIR/$rid.json"
    if [[ -f "$r" ]] && grep -q '"status": "ok"' "$r" 2>/dev/null; then continue; fi
    PENDING+=("$c")
done
TOTAL=$(ls "$CONFIG_DIR"/*.json | wc -l)
if [[ ${#PENDING[@]} -eq 0 ]]; then
    echo "All $TOTAL configs complete."
    exit 0
fi
echo "Pending: ${#PENDING[@]} of $TOTAL configs"
printf '%s\n' "${PENDING[@]}" > "$OUT_DIR/_pending.txt"

# ---- warm caches for encoders that need it ------------------------------ #
NEED=()
for fm in $(for c in "${PENDING[@]}"; do
                python3 -c "import json;print(json.load(open('$c'))['fm'])"
            done | sort -u); do
    # A cache is warm only if it exists for the max_slides THIS SWEEP uses.
    # Hardcoding _n600 here silently reported "warm" after max_slides changed, and
    # the run jobs then died inside the 7-minute per-config timeout while building
    # caches. Derive the suffix from the configs instead.
    missing=0
    for ms in $(for c in "${PENDING[@]}"; do
                    python3 -c "import json;m=json.load(open('$c'));print(m['max_slides'] if m.get('max_slides') else 'all')"
                done | sort -u); do
        for ds in TCGA-LUNG PANDA BRACS UBC-OCEAN BACH; do
            [[ -f "$CACHE_DIR/${ds}_${fm}_p256_n${ms}.npz" ]] || missing=1
        done
    done
    [[ $missing -eq 1 ]] && NEED+=("$fm")
done

if [[ ${#NEED[@]} -gt 0 ]]; then
    echo "Warming caches for ${#NEED[@]} encoder(s): ${NEED[*]}"
    for fm in "${NEED[@]}"; do
        wait_for_slot
        jid=$(sbatch --parsable --job-name="warm_${fm#features_}" \
            --output="logs/warm_${fm#features_}_%j.log" \
            --account="$ACCOUNT" --partition="$PARTITION" \
            --mem=96G --cpus-per-task=16 --gpus=1 --nodes=1 --time="$WALLTIME" \
            --wrap="$MODULE_CMD && for C in \$(grep -l '\"fm\": \"$fm\"' $CONFIG_DIR/*.json); do \
                      python -u scripts/run_config.py --config \$C \
                        --cache_dir $CACHE_DIR --cache_only || true ; \
                    done")
        echo "  $fm -> job $jid"
        sleep 5
    done
else
    echo "All caches already warm."
fi

# ---- chunked run jobs --------------------------------------------------- #
# ~30s per config (6 targets x 10 controls x 2 probes). Chunk so each job fits
# comfortably inside the walltime, and throttle to the QOS submit limit.
CHUNK="${CHUNK:-20}"   # ~2 min/config -> ~40 min/chunk
split -l "$CHUNK" -d "$OUT_DIR/_pending.txt" "$OUT_DIR/_chunk_"

RUN_IDS=()
for chunk in "$OUT_DIR"/_chunk_*; do
    n=$(wc -l < "$chunk")
    wait_for_slot
    jid=$(sbatch --parsable --job-name=purge_sweep \
        --output="logs/purge_sweep_%j.log" \
        --account="$ACCOUNT" --partition="$PARTITION" \
        --mem=96G --cpus-per-task=16 --gpus=1 --nodes=1 --time="$WALLTIME" \
        --wrap="$MODULE_CMD && \
                while read -r CFG; do \
                  timeout 1800 python -u scripts/run_config.py --config \"\$CFG\" \
                    --out_dir $OUT_DIR --cache_dir $CACHE_DIR || \
                    echo \"TIMEOUT/FAIL \$CFG\" ; \
                done < $chunk")
    RUN_IDS+=("$jid")
    echo "  chunk $(basename "$chunk"): $n configs -> job $jid"
    sleep 5
done

echo
echo "Run jobs: ${RUN_IDS[*]}"
echo "Collect:  python scripts/collect_results.py"
