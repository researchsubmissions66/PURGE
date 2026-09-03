#!/bin/bash
# One MIL task per job, all four GPUs to that single run.
#
# TCGA-LUNG slides average ~17,800 patches, so a run that shares a node with
# another finishes only a few epochs inside the 55-minute interactive cap.
# ABMIL and MeanMIL both completed solo in ~27 minutes, so give them the node.
set -uo pipefail
cd "$(dirname "$0")/.."
TASKS=${1:-configs/mil_tcga_feasible.txt}
TAG=${2:-t}
ACCOUNT=bhwm-delta-gpu
PART=gpuA40x4-interactive,gpuA100x4-interactive
MOD="module load pytorch-conda"

n=0
while read -r t; do
    [[ -z "$t" ]] && continue
    n=$((n+1))
    out=$(sed -n 's/.*--output_json \([^ ]*\).*/\1/p' <<<"$t")
    [[ -f "$out" ]] && { echo "  skip $(basename "$out")"; continue; }
    jid=""
    for attempt in $(seq 1 240); do
        jid=$(sbatch --parsable --job-name="milsolo${TAG}_${n}" \
              --output="logs/milsolo${TAG}_${n}_%j.log" \
              --account=$ACCOUNT --partition=$PART \
              --mem=200G --cpus-per-task=32 --gpus=4 --nodes=1 --time="00:55:00" \
              --wrap="$MOD && timeout 3200 $t" 2>/dev/null)
        [[ -n "$jid" ]] && break
        sleep 45
    done
    [[ -z "$jid" ]] && { echo "  !! could not submit task $n"; continue; }
    echo "  task $n -> $jid  $(basename "$out")"
    while squeue -h -j "$jid" -o "%t" 2>/dev/null | grep -q .; do sleep 30; done
done < "$TASKS"
echo "[$TAG] SOLO DRIVER DONE"
