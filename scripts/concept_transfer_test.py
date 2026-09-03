"""
A practitioner's check: is this "concept direction" real, or cohort geometry?

MOTIVATION
----------
A linear probe on a pathology foundation model scores 0.95 for grade or subtype,
and that is routinely read as the encoder ENCODING the concept. Our sweep shows
the probe can be reading a subspace specific to the cohort it was fitted on: an
eraser fitted on BACH leaves BRACS atypia untouched at 0.898, though both are
breast, and the five weakest configurations in the whole sweep are cross-cohort
runs where collateral exceeds target damage.

This script turns that into a test anyone can run before making a capability
claim, without needing the erasure machinery.

THE TEST
--------
Fit a linear probe for the concept on cohort A. Then:

  within   probe A -> held-out patients of A      (the number usually reported)
  transfer probe A -> cohort B, same concept      (what the claim implies)
  refit    probe B -> held-out patients of B      (B's own ceiling)

The diagnostic is the TRANSFER GAP, within minus transfer, measured against
refit. A concept the encoder genuinely represents should transfer: the direction
means the same thing in both cohorts. A direction that is cohort geometry will
score well within and collapse across, while refit stays high, proving the
information is present in B and only the DIRECTION failed to carry over.

Reporting `within` alone cannot distinguish these two cases, which is the
methodological point of the paper.

USAGE
-----
    python scripts/concept_transfer_test.py \
        --cohort_a BACH --cohort_b BRACS --concept malignancy
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.splits import patient_folds                        # noqa: E402

VERDICT = [
    (0.10, "TRANSFERS. The direction means the same thing in both cohorts; a "
           "capability claim is supported."),
    (0.25, "PARTIAL. Some of what the probe reads is shared, some is cohort-"
           "specific. Report both numbers."),
    (1.01, "DOES NOT TRANSFER. The probe is largely reading cohort geometry. A "
           "capability claim from the within-cohort number alone is not "
           "supported by this evidence."),
]


def load(ds, fm, cache_dir):
    p = f"{cache_dir}/{ds}_{fm}_p256_nall.npz"
    if not os.path.exists(p):
        p = f"{cache_dir}/{ds}_{fm}_p256_n2000.npz"
    if not os.path.exists(p):
        raise SystemExit(f"no cached features for {ds} under {cache_dir}")
    z = np.load(p, allow_pickle=True)
    return np.asarray(z['X'], dtype=np.float32), z['labels'], z['patients']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cohort_a', required=True)
    ap.add_argument('--cohort_b', required=True)
    ap.add_argument('--labels_a', nargs='+', required=True,
                    help="the two class labels in cohort A, negative first")
    ap.add_argument('--labels_b', nargs='+', required=True,
                    help="the matching two class labels in cohort B")
    ap.add_argument('--fm', default='features_virchow2')
    ap.add_argument('--cache_dir', default='results/quick/cache')
    ap.add_argument('--metadata', default='data/multi_benchmark_metadata.csv')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def prep(ds, labels):
        X, y, pat = load(ds, a.fm, a.cache_dir)
        keep = np.isin(y, labels)
        X, y, pat = X[keep], y[keep], pat[keep]
        yb = (y == labels[1]).astype(int)
        tr_p, te_p = patient_folds(a.metadata, ds, a.fold)
        return X, yb, np.isin(pat, tr_p), np.isin(pat, te_p)

    XA, yA, trA, teA = prep(a.cohort_a, a.labels_a)
    XB, yB, trB, teB = prep(a.cohort_b, a.labels_b)
    print(f"{a.cohort_a}: {trA.sum()} train / {teA.sum()} test    "
          f"{a.cohort_b}: {trB.sum()} train / {teB.sum()} test")

    pa = LogisticRegression(max_iter=3000).fit(XA[trA], yA[trA])
    pb = LogisticRegression(max_iter=3000).fit(XB[trB], yB[trB])
    within   = roc_auc_score(yA[teA], pa.predict_proba(XA[teA])[:, 1])
    transfer = roc_auc_score(yB[teB], pa.predict_proba(XB[teB])[:, 1])
    refit    = roc_auc_score(yB[teB], pb.predict_proba(XB[teB])[:, 1])
    # Orientation is arbitrary across cohorts; a flipped direction is still a
    # direction that carried over.
    transfer = max(transfer, 1 - transfer)
    gap = within - transfer

    cos = float(pa.coef_[0] @ pb.coef_[0] /
                (np.linalg.norm(pa.coef_[0]) * np.linalg.norm(pb.coef_[0]) + 1e-12))

    print(f"\n  within   {a.cohort_a} -> {a.cohort_a}   {within:.4f}   <- the number usually reported")
    print(f"  transfer {a.cohort_a} -> {a.cohort_b}   {transfer:.4f}   <- what the claim implies")
    print(f"  refit    {a.cohort_b} -> {a.cohort_b}   {refit:.4f}   <- B's own ceiling")
    print(f"\n  transfer gap {gap:+.4f}     probe direction cosine {cos:+.4f}")
    verdict = next(v for t, v in VERDICT if gap < t)
    print(f"\n  {verdict}")
    if transfer < 0.6 <= refit:
        print("  Note: B's own probe scores well, so the information IS present in B. "
              "Only the direction failed to carry over.")

    if a.out:
        json.dump(dict(within=within, transfer=transfer, refit=refit, gap=gap,
                       cosine=cos, verdict=verdict, cohort_a=a.cohort_a,
                       cohort_b=a.cohort_b, fm=a.fm, fold=a.fold),
                  open(a.out, 'w'), indent=2)
        print(f"  wrote {a.out}")


if __name__ == '__main__':
    main()
