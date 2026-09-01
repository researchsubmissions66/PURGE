"""
Paired significance test: does erasure degrade the target more than the controls?

For each target dataset, compares per-fold AUC of the baseline probe against the
probe retrained through the eraser, using a paired t-test and a Wilcoxon signed-
rank test over folds.

With the usual 5 folds these tests have very little power. Treat a significant
result as supporting evidence, never as the headline, and always report the
effect size (the AUC drop) alongside the p-value.
"""

import argparse
from collections import defaultdict

import numpy as np
from scipy import stats

from aggregate_results import collect


def paired_tests(baseline, erased):
    """Returns (n, mean_drop, t_p, w_p). p-values are None when undefined."""
    n = min(len(baseline), len(erased))
    if n == 0:
        return 0, float('nan'), None, None
    b, e = np.asarray(baseline[:n]), np.asarray(erased[:n])
    drop = b - e
    mean_drop = float(drop.mean())

    t_p = w_p = None
    if n >= 2 and np.ptp(drop) > 0:
        t_p = float(stats.ttest_rel(b, e).pvalue)
    if n >= 6 and np.ptp(drop) > 0:
        w_p = float(stats.wilcoxon(b, e).pvalue)
    return n, mean_drop, t_p, w_p


def fmt_p(p):
    if p is None:
        return "n/a"
    return f"{p:.4f}" + ("*" if p < 0.05 else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results_dir', default='results/sweep')
    args = parser.parse_args()

    baselines, evals, legacy = collect(args.results_dir)
    if not evals:
        raise SystemExit(f"No cross-evaluation results under {args.results_dir}")
    if legacy:
        print("NOTE: some runs only recorded legacy 'best_val_auc'; those are "
              "optimistically biased.\n")

    for fm in sorted(evals):
        base = baselines.get(fm, {})
        if not base:
            print(f"## {fm}: no baseline runs found, skipping\n")
            continue

        pairs = evals[fm]
        targets = sorted({t for t, _ in pairs})

        print(f"## Paired tests - {fm}\n")
        print("| Target | Comparison | n folds | mean AUC drop | t-test p | Wilcoxon p |")
        print("|--------|------------|---------|---------------|----------|------------|")

        for target in targets:
            if (target, target) not in pairs or target not in base:
                continue
            n, drop, t_p, w_p = paired_tests(base[target], pairs[(target, target)])
            print(f"| {target} | target (self) | {n} | {drop:+.4f} | "
                  f"{fmt_p(t_p)} | {fmt_p(w_p)} |")

            collateral = defaultdict(list)
            for (t, c), vals in pairs.items():
                if t == target and c != target and c in base:
                    n_c = min(len(base[c]), len(vals))
                    collateral['b'].extend(base[c][:n_c])
                    collateral['e'].extend(vals[:n_c])
            if collateral:
                n, drop, t_p, w_p = paired_tests(collateral['b'], collateral['e'])
                print(f"| {target} | controls (pooled) | {n} | {drop:+.4f} | "
                      f"{fmt_p(t_p)} | {fmt_p(w_p)} |")
        print()


if __name__ == "__main__":
    main()
