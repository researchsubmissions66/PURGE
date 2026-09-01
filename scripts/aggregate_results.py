"""
Aggregate sweep results into a baseline table and a target x control matrix.

Reads results/sweep/<fm>/{baseline_*,eval_*}.json produced by train_mil.py.
Prefers the held-out 'test_auc'; falls back to legacy 'best_val_auc' and marks
those rows, because that legacy field was a max-over-epochs value selected on the
same split it was measured on and is optimistically biased.

Chance level for macro one-vs-rest AUC is 0.50 for any number of classes.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

BASELINE_RE = re.compile(r'^baseline_(?P<dataset>.+)_fold(?P<fold>\d+)\.json$')
EVAL_RE = re.compile(
    r'^eval_target-(?P<target>.+?)_control-(?P<control>.+?)_fold(?P<fold>\d+)\.json$'
)


def read_auc(path):
    """Returns (auc, is_legacy) or (None, _) when the run recorded no usable AUC."""
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, False
    if payload.get('test_auc') is not None:
        return float(payload['test_auc']), False
    if payload.get('best_val_auc') is not None:
        return float(payload['best_val_auc']), True
    return None, False


def collect(results_dir):
    baselines = defaultdict(lambda: defaultdict(list))   # fm -> dataset -> [auc]
    evals = defaultdict(lambda: defaultdict(list))       # fm -> (target, control) -> [auc]
    legacy = False

    for path in glob.glob(os.path.join(results_dir, '**', '*.json'), recursive=True):
        name = os.path.basename(path)
        fm = os.path.basename(os.path.dirname(path))
        auc, is_legacy = read_auc(path)
        if auc is None:
            continue
        legacy |= is_legacy

        m = BASELINE_RE.match(name)
        if m:
            baselines[fm][m.group('dataset')].append(auc)
            continue
        m = EVAL_RE.match(name)
        if m:
            evals[fm][(m.group('target'), m.group('control'))].append(auc)

    return baselines, evals, legacy


def fmt(values):
    if not values:
        return "  --  "
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{np.mean(values):.3f}+-{np.std(values):.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results_dir', default='results/sweep')
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        raise SystemExit(f"No results directory at {args.results_dir}")

    baselines, evals, legacy = collect(args.results_dir)
    if not baselines and not evals:
        raise SystemExit(f"No parseable result JSON under {args.results_dir}")

    if legacy:
        print("NOTE: some runs only recorded 'best_val_auc' (max-over-epochs on the "
              "evaluation split). Those numbers are optimistically biased; re-run "
              "them with the current train_mil.py to get held-out test AUC.\n")

    for fm in sorted(baselines):
        print(f"## Baseline AUC - {fm}\n")
        print("| Dataset | AUC | folds |")
        print("|---------|-----|-------|")
        for dataset in sorted(baselines[fm]):
            vals = baselines[fm][dataset]
            print(f"| {dataset} | {fmt(vals)} | {len(vals)} |")
        print()

    for fm in sorted(evals):
        pairs = evals[fm]
        targets = sorted({t for t, _ in pairs})
        controls = sorted({c for _, c in pairs})

        print(f"## Cross-evaluation - {fm}")
        print("Rows: dataset the eraser targeted. Columns: dataset the fresh probe "
              "was trained on.")
        print("The diagonal should collapse toward 0.50; off-diagonals should not.\n")

        header = "| Target \\ Control | " + " | ".join(controls) + " |"
        print(header)
        print("|" + "---|" * (len(controls) + 1))

        for target in targets:
            cells = []
            for control in controls:
                vals = pairs.get((target, control), [])
                cell = fmt(vals)
                if target == control and vals:
                    cell = f"**{cell}**"
                cells.append(cell)
            print(f"| {target} | " + " | ".join(cells) + " |")
        print()

        base = baselines.get(fm, {})
        rows = []
        for target in targets:
            diag = pairs.get((target, target), [])
            if not diag or target not in base:
                continue
            target_drop = np.mean(base[target]) - np.mean(diag)
            collateral = [
                np.mean(base[c]) - np.mean(pairs[(target, c)])
                for c in controls
                if c != target and (target, c) in pairs and c in base
            ]
            if not collateral:
                continue
            sds = target_drop / (1e-3 + np.mean(np.abs(collateral)))
            rows.append((target, target_drop, float(np.mean(collateral)), sds))

        if rows:
            print("### Selective degradation\n")
            print("| Target | target drop | mean collateral | SDS |")
            print("|--------|-------------|-----------------|-----|")
            for target, drop, coll, sds in rows:
                print(f"| {target} | {drop:+.3f} | {coll:+.3f} | {sds:.2f} |")
            print()


if __name__ == "__main__":
    main()
