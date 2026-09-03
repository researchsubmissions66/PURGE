"""
Fit MIL erasers the way the pooled sweep does: on the fold's training-split
SLIDE MEANS, not on patch embeddings.

Why this script exists
----------------------
`fit_unlearner.py` samples PATCH embeddings and takes their top-k variance
directions. Patch covariance is dominated by within-slide texture, stain and
position, which is not where slide-level label signal lives, so the resulting
subspace barely touches a slide-level task. Measured on TCGA-LUNG fold 0
(logreg on slide means, clean 0.9540):

    patch-fit, 300 slides    0.7975
    patch-fit, 1000 slides   0.9172      <- fitting it BETTER makes it WEAKER
    slide-mean fit           0.5355      <- what the pooled sweep reports

Overlap between a patch-fit and a slide-mean-fit subspace is only 0.698 mean
principal cosine: different subspaces, not noisy versions of one.

Erasure and mean pooling are both linear and commute, so a slide-mean-fitted U
applied at patch level makes MeanMIL reproduce the pooled number exactly. That
gives the MIL sweep an anchor it has never had, and lets ABMIL/TransMIL measure
architectural resistance to the SAME attack the rest of the paper reports.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.unlearning.subspace import svd_subspace          # noqa: E402
from src.utils.splits import patient_folds                # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+',
                    default=['TCGA-LUNG', 'BRACS', 'PANDA'])
    ap.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--fm', default='features_virchow2')
    ap.add_argument('--cache_dir', default='results/quick/cache')
    ap.add_argument('--metadata', default='data/multi_benchmark_metadata.csv')
    ap.add_argument('--out_dir', default='results/unlearners_slide')
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    for ds in a.datasets:
        cache = f"{a.cache_dir}/{ds}_{a.fm}_p256_nall.npz"
        if not os.path.exists(cache):
            print(f"!! no cache for {ds}: {cache}")
            continue
        z = np.load(cache, allow_pickle=True)
        X = torch.tensor(z['X']).float()
        pats = z['patients']
        for fold in a.folds:
            out = f"{a.out_dir}/{ds}_fold{fold}_k{a.k}.pt"
            if os.path.exists(out):
                print(f"  {os.path.basename(out)} present")
                continue
            # OUTER train only. The fold's held-out patients are the victim's
            # test set and the attacker must not see them.
            train_p, _ = patient_folds(a.metadata, ds, fold)
            m = torch.tensor(np.isin(pats, train_p))
            Xtr = X[m]
            U = svd_subspace(Xtr, k=a.k)
            mu = Xtr.mean(0)
            torch.save({'U': U, 'mu': mu}, out)
            print(f"  {ds} fold{fold}: fit on {int(m.sum())} train slides -> "
                  f"U{tuple(U.shape)} -> {out}")


if __name__ == '__main__':
    main()
