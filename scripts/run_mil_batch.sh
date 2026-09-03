#!/bin/bash
# Run a slice of configs/mil_tasks.txt, 4 at a time across the node's 4 GPUs.
#
# MIL training is ~27 min and I/O-bound on h5 reads, so one run does not saturate
# a 4-GPU node. Packing four per job turns 82 submissions into a handful, which
# matters because the submit-limit is shared with an unrelated 383-job workload.
set -uo pipefail
cd "$(dirname "$0")/.."
TASKS="${1:-configs/mil_tasks.txt}"
START="${2:-0}"
COUNT="${3:-14}"
NGPU="${NGPU:-4}"

mapfile -t ALL < "$TASKS"
SLICE=("${ALL[@]:$START:$COUNT}")
echo "running ${#SLICE[@]} tasks (offset $START) ${NGPU}-way parallel"

i=0
for t in "${SLICE[@]}"; do
    [[ -z "$t" ]] && continue
    # Skip work another chunk already finished (jobs may overlap on restart).
    out=$(sed -n 's/.*--output_json \([^ ]*\).*/\1/p' <<<"$t")
    if [[ -n "$out" && -f "$out" ]]; then echo "SKIP $(basename "$out")"; continue; fi

    gpu=$(( i % NGPU )); i=$(( i + 1 ))
    ( CUDA_VISIBLE_DEVICES=$gpu timeout 7200 $t \
        > "logs/milb_$(basename "${out%.json}").log" 2>&1 \
        && echo "OK $(basename "$out")" || echo "FAIL $(basename "$out")" ) &

    # Throttle to NGPU concurrent; wait -n returns as soon as any one finishes.
    while (( $(jobs -rp | wc -l) >= NGPU )); do wait -n; done
done
wait
echo "chunk done"
