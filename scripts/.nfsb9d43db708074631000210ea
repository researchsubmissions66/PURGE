#!/bin/bash
# Robust sequential driver for the sweep: warm caches, then run configs.
#
# Submits ONE job at a time and waits for it to finish before the next. Slower
# than parallel, but the interactive QOS allows only 2 submitted jobs and its
# accounting lags submission - an optimistic "is there a slot?" check races and
# fails with QOSMaxSubmitJobPerUserLimit. Retrying on failure is the reliable
# pattern; predicting availability is not.
set -uo pipefail
cd "$(dirname "$0")/.."

CONFIG_DIR=${CONFIG_DIR:-configs/sweeps}
OUT_DIR=${OUT_DIR:-results/sweep_v2}
CACHE_DIR=${CACHE_DIR:-results/quick/cache}
ACCOUNT=bhwm-delta-gpu
PART=gpuA40x4-interactive,gpuA100x4-interactive
MOD="module load pytorch-conda"

submit_and_wait() {   # name walltime command...
    local name=$1 wt=$2; shift 2
    local jid=""
    # Patience, not coordination: other runners (e.g. the MIL confirmation) may
    # legitimately hold both interactive slots for an hour or more. 10 retries was
    # ~10 minutes and gave up while MIL jobs were mid-run.
    for attempt in $(seq 1 240); do
        jid=$(sbatch --parsable --job-name="$name" --output="logs/${name}_%j.log" \
              --account=$ACCOUNT --partition=$PART \
              --mem=96G --cpus-per-task=16 --gpus=1 --nodes=1 --time="$wt" \
              --wrap="$MOD && $*" 2>/dev/null)
        [[ -n "$jid" ]] && break
        sleep 45          # QOS full or lagging; back off and retry
    done
    [[ -z "$jid" ]] && { echo "  !! could not submit $name"; return 1; }
    echo "  $name -> $jid"
    while squeue -h -j "$jid" -o "%t" 2>/dev/null | grep -q .; do sleep 30; done
    return 0
}

# ---- 1. warm every (encoder, max_slides) combination the sweep needs ------ #
echo "== cache warming =="
python3 - > /tmp/_warm_plan.txt <<'PY'
import json, glob, os
need = set()
for f in glob.glob('configs/sweeps/*.json'):
    c = json.load(open(f))
    ms = c['max_slides'] if c.get('max_slides') else None
    tag = str(ms) if ms else 'all'
    # Only require a cache for cohorts that HAVE features for this encoder.
    # conch_v15 and virchow have zero BACH slides, so demanding a BACH cache for
    # them makes the plan permanently unsatisfiable and re-warms every restart.
    def has_features(ds, fm):
        for px in (224, 256, 512):
            root = ('/work/hdd/bhwm/trident_features/master_benchmark'
                    if ds.startswith('TCGA') else f'/work/hdd/bhwm/{ds}')
            d = f'{root}/20x_{px}px_0px_overlap/{fm}'
            if os.path.isdir(d) and any(x.endswith('.h5') for x in os.listdir(d)):
                return True
        return False

    ok = all(os.path.exists(f"results/quick/cache/{ds}_{c['fm']}_p256_n{tag}.npz")
             for ds in ('TCGA-LUNG', 'PANDA', 'BRACS', 'UBC-OCEAN', 'BACH')
             if has_features(ds, c['fm']))
    if not ok:
        need.add((c['fm'], tag, f))
# One warm job per (encoder, max_slides), not per config: several configs share
# a cache, and submitting one job each wastes scarce interactive slots on no-ops.
rep = {}
for fm, tag, f in need:
    rep.setdefault((fm, tag), f)
for (fm, tag), f in sorted(rep.items()):
    print(f"{fm} {tag} {f}")
PY
while read -r fm tag cfg; do
    [[ -z "$fm" ]] && continue
    submit_and_wait "warm_${fm#features_}_${tag}" "00:55:00" \
        "python -u scripts/run_config.py --config $cfg --cache_dir $CACHE_DIR --cache_only"
done < /tmp/_warm_plan.txt

# ---- 2. run pending configs in chunks ------------------------------------ #
echo "== running configs =="
while :; do
    PEND=()
    for c in "$CONFIG_DIR"/*.json; do
        rid=$(basename "$c" .json)
        r="$OUT_DIR/$rid.json"
        [[ -f "$r" ]] && grep -q '"status": "ok"' "$r" 2>/dev/null && continue
        PEND+=("$c")
    done
    [[ ${#PEND[@]} -eq 0 ]] && { echo "all configs complete"; break; }
    echo "  ${#PEND[@]} pending"
    printf '%s\n' "${PEND[@]:0:40}" > "$OUT_DIR/_batch.txt"
    submit_and_wait "purge_sweep" "00:55:00" \
        "bash scripts/run_batch.sh $OUT_DIR/_batch.txt $OUT_DIR $CACHE_DIR"
    n=$(ls "$OUT_DIR"/*.json 2>/dev/null | grep -v '/_' | wc -l)
    echo "  completed: $n"
    # Stop if a pass makes no progress. Some configs fail deterministically
    # (method-splince: "Could only find 0 orthogonal vectors" at n<<d), and a
    # plain "until pending is empty" loop would resubmit them forever.
    if [[ "${#PEND[@]}" -eq "${LAST_PEND:-0}" ]]; then
        echo "  no progress this pass (${#PEND[@]} permanently failing) - stopping"
        for c in "${PEND[@]}"; do echo "    unresolved: $(basename "$c")"; done
        break
    fi
    LAST_PEND=${#PEND[@]}
done
echo "DRIVER DONE"
