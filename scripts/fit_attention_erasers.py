"""
Fit erasers that target ATTENTION pooling, not just mean pooling.

THE MECHANISM
-------------
Plain affine SVD removes a fixed subspace U from every patch. Mean pooling sees
only the slide mean, which loses exactly span(U), so MeanMIL collapses. Attention
SELECTS patches rather than averaging them, so it can still read signal carried
in the within-slide deviations z_j - mean(z). Plain SVD never touches those.

Measured on BRACS, 7-class, 5 folds, identical erasers and slides:

    MeanMIL   drop +0.3385
    ABMIL     drop +0.0555     6.1x resistance
    TransMIL  drop +0.0201    16.8x resistance

WHAT THIS SCRIPT ADDS
---------------------
`U_dev`, a subspace fitted on the within-slide deviations pooled across training
slides. Removing span(U_mean) + span(U_dev) is still ONE fixed linear projector
applied identically to every patch, so it remains directly comparable with plain
SVD and cannot be waved away as an adaptive or bag-dependent transform. It simply
also deletes the directions along which patches within a slide differ from each
other, which is the resource attention exploits.

`alpha`, for the svd_bag variant, shrinks whatever within-slide spread survives.
alpha = 1 is plain SVD; alpha = 0 makes every patch identical to the erased slide
mean, at which point attention over identical patches IS mean pooling and no
pooling rule can recover what the mean lost. alpha traces a strength against
detectability curve rather than being a single recommended setting: at alpha = 0
the bag is visibly degenerate, and that cost should be reported, not hidden.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.fit_unlearner import build_index                     # noqa: E402
from src.datasets.feature_dataset import to_h5_name               # noqa: E402
from src.unlearning.subspace import svd_subspace                  # noqa: E402
from src.utils.splits import patient_folds                        # noqa: E402

ROOT = '/work/hdd/bhwm'
V = '20x_224px_0px_overlap/features_virchow2'
DIRS = {
    'TCGA-LUNG': f'{ROOT}/trident_features/master_benchmark/{V}',
    'PANDA': f'{ROOT}/PANDA/{V}',
    'BRACS': f'{ROOT}/BRACS/{V}',
}


def deviations(df, index, max_slides, patches, block, seed):
    """Within-slide patch deviations, pooled across slides. Returns [N, D]."""
    rng = np.random.RandomState(seed)
    if len(df) > max_slides:
        df = df.sample(max_slides, random_state=seed)
    out, missing = [], 0
    for _, row in df.iterrows():
        p = index.get(to_h5_name(row['filename']))
        if p is None:
            missing += 1
            continue
        try:
            with h5py.File(p, 'r') as f:
                feats = f['features']
                n = feats.shape[0]
                if n > patches:
                    nb = max(1, patches // block)
                    size = max(1, patches // nb)
                    starts = np.unique(np.linspace(0, max(n - size, 0), nb).astype(int))
                    z = np.concatenate([feats[s:s + size] for s in starts], 0)
                else:
                    z = feats[:]
        except (OSError, KeyError):
            missing += 1
            continue
        z = torch.tensor(np.asarray(z, dtype=np.float32))
        if z.shape[0] < 2:
            continue
        out.append(z - z.mean(0, keepdim=True))     # the part attention reads
    if missing:
        print(f"    ({missing} slides unreadable)")
    return torch.cat(out, 0) if out else torch.empty(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=['BRACS'])
    ap.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--k', type=int, default=64, help="rank of the mean subspace")
    ap.add_argument('--k_dev', type=int, default=64, help="rank of the deviation subspace")
    ap.add_argument('--alpha', type=float, nargs='+', default=[0.0, 0.5],
                    help="within-slide shrink factors for the svd_bag variant")
    ap.add_argument('--max_slides', type=int, default=800)
    ap.add_argument('--patches', type=int, default=64)
    ap.add_argument('--block', type=int, default=32)
    ap.add_argument('--fm', default='features_virchow2')
    ap.add_argument('--cache_dir', default='results/quick/cache')
    ap.add_argument('--metadata', default='data/multi_benchmark_metadata.csv')
    ap.add_argument('--slide_dir', default='results/unlearners_slide')
    ap.add_argument('--out_dir', default='results/unlearners_attn')
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    md = pd.read_csv(a.metadata)

    for ds in a.datasets:
        if ds not in DIRS or not os.path.isdir(DIRS[ds]):
            print(f"!! no feature root for {ds}")
            continue
        index = build_index([DIRS[ds]])
        sub = md[md.dataset == ds]
        for fold in a.folds:
            base = f"{a.slide_dir}/{ds}_fold{fold}_k{a.k}.pt"
            if not os.path.exists(base):
                print(f"!! missing slide eraser {base}")
                continue
            w = torch.load(base, weights_only=False)
            train_p, _ = patient_folds(a.metadata, ds, fold)
            tr = sub[sub.patient_id.isin(train_p)]

            out_dev = f"{a.out_dir}/{ds}_fold{fold}_dev.pt"
            if not os.path.exists(out_dev):
                print(f"  {ds} fold{fold}: pooling deviations from "
                      f"{min(len(tr), a.max_slides)} train slides")
                D = deviations(tr, index, a.max_slides, a.patches, a.block, fold)
                if D.numel() == 0:
                    print("    no deviations read, skipping")
                    continue
                # Deviations are already centred per slide, so their principal
                # directions are the within-slide axes of variation.
                U_dev = svd_subspace(D, k=a.k_dev)
                torch.save({'U': w['U'], 'mu': w['mu'], 'U_dev': U_dev}, out_dev)
                print(f"    deviations {tuple(D.shape)} -> U_dev {tuple(U_dev.shape)}"
                      f" -> {out_dev}")
            else:
                print(f"  {out_dev} present")

            for al in a.alpha:
                out_bag = f"{a.out_dir}/{ds}_fold{fold}_bag{al}.pt"
                if os.path.exists(out_bag):
                    continue
                torch.save({'U': w['U'], 'mu': w['mu'], 'alpha': al}, out_bag)
                print(f"    alpha={al} -> {out_bag}")


if __name__ == '__main__':
    main()
