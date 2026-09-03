#!/bin/bash
# Drive a RANGE of configs/mil_tasks.txt through the interactive partition.
#
#   bash scripts/mil_range_driver.sh <start> <end> <tag>
#
# Two instances on disjoint ranges use both interactive slots without racing.
# CHUNK=2 on 2 GPUs, not 4 on 4: a 4-way chunk finished only 1 of 4 tasks inside
# the 55-minute cap, because four concurrent MIL runs contend on h5 reads badly
# enough to more than double each run's ~27 minutes.
set -uo pipefail
cd "$(dirname "$0")/.."
TASKS=configs/mil_tasks.txt
START=${1:-0}; END=${2:-999}; TAG=${3:-a}
CHUNK=2
ACCOUNT=bhwm-delta-gpu
PART=gpuA40x4-interactive,gpuA100x4-interactive
MOD="module load pytorch-conda"

submit_and_wait() {
    local name=$1 wt=$2; shift 2
    local jid=""
    for attempt in $(seq 1 240); do
        jid=$(sbatch --parsable --job-name="$name" --output="logs/${name}_%j.log" \
              --account=$ACCOUNT --partition=$PART \
              --mem=120G --cpus-per-task=16 --gpus=2 --nodes=1 --time="$wt" \
              --wrap="$MOD && $*" 2>/dev/null)
        [[ -n "$jid" ]] && break
        sleep 45
    done
    [[ -z "$jid" ]] && { echo "  !! could not submit $name"; return 1; }
    echo "  $name -> $jid"
    while squeue -h -j "$jid" -o "%t" 2>/dev/null | grep -q .; do sleep 30; done
}

total=$(grep -c . "$TASKS")
(( END > total )) && END=$total
off=$START
while (( off < END )); do
    n=$CHUNK; (( off + n > END )) && n=$(( END - off ))
    # skip a chunk whose outputs all exist already
    missing=0
    for i in $(seq 0 $((n-1))); do
        t=$(sed -n "$((off+i+1))p" "$TASKS")
        [[ -z "$t" ]] && continue
        o=$(sed -n 's/.*--output_json \([^ ]*\).*/\1/p' <<<"$t")
        [[ -f "$o" ]] || missing=$((missing+1))
    done
    if (( missing == 0 )); then off=$(( off + CHUNK )); continue; fi
    echo "  [$TAG] offset=$off n=$n ($missing missing)"
    NGPU=2 submit_and_wait "milr${TAG}_${off}" "00:55:00" \
        "NGPU=2 bash scripts/run_mil_batch.sh $TASKS $off $n"
    off=$(( off + CHUNK ))
done
echo "[$TAG] RANGE DONE"
