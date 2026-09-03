#!/bin/bash
# Run every config listed in $1. A real script, not an sbatch --wrap one-liner:
# the previous version nested four levels of quoting (heredoc -> --wrap -> inner
# shell) and "$C" reached python as a literal string, so a whole batch no-opped.
set -uo pipefail
cd "$(dirname "$0")/.."
BATCH="$1"
OUT="${2:-results/sweep_v2}"
CACHE="${3:-results/quick/cache}"

n=0; ok=0
while read -r cfg; do
    [[ -z "$cfg" ]] && continue
    n=$((n+1))
    if timeout 1500 python -u scripts/run_config.py \
            --config "$cfg" --out_dir "$OUT" --cache_dir "$CACHE"; then
        ok=$((ok+1))
    else
        echo "SKIP $cfg (exit $?)"
    fi
done < "$BATCH"
echo "batch done: $ok/$n succeeded"
