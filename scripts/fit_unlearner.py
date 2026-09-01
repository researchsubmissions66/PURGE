"""
Fit a closed-form eraser (SVD null-space, LEACE, or SPLINCE) for one concept.

Two fixes over the earlier version worth knowing about:

* Features are sampled at PATCH level. The eraser is applied to patch embeddings
  during MIL training, so fitting it on slide-mean vectors fits the wrong
  distribution - slide means are far less dispersed than the patches they pool.

* Affine erasers are stored in the canonical convention x' = x @ P.T + b with
  b = mu - P @ mu (see src/unlearning/apply.py). Storing the raw mean as b and
  applying x @ P.T + mu leaves a residual offset of mu @ P.T that silently
  corrupts the erasure.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.datasets.feature_dataset import to_h5_name
from src.unlearning.apply import save_affine
from src.unlearning.subspace import discriminative_subspace, svd_subspace

DEFAULT_METADATA = 'data/multi_benchmark_metadata.csv'


def load_patch_features(df, encoder_dir, max_slides, patches_per_slide, seed=0):
    """Sample patch embeddings from a cohort. Returns [N, D]."""
    rng = np.random.RandomState(seed)
    if len(df) > max_slides:
        df = df.sample(max_slides, random_state=seed)

    chunks, missing = [], 0
    for _, row in df.iterrows():
        path = os.path.join(encoder_dir, to_h5_name(row['filename']))
        try:
            with h5py.File(path, 'r') as f:
                feats = f['features']
                n = feats.shape[0]
                if n > patches_per_slide:
                    idx = np.sort(rng.choice(n, patches_per_slide, replace=False))
                    chunks.append(torch.tensor(feats[idx]))
                else:
                    chunks.append(torch.tensor(feats[:]))
        except (OSError, KeyError):
            missing += 1

    if missing:
        print(f"  ({missing} of {len(df)} slides unreadable under {encoder_dir})")
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks, dim=0).float()


def fit_svd(X_pos, X_neg, k, discriminative):
    if discriminative:
        print(f"Fitting discriminative subspace (k={k})...")
        return discriminative_subspace(X_pos, X_neg, k=k)
    print(f"Fitting variance subspace of the target cohort (k={k})...")
    return svd_subspace(X_pos, k=k)


def fit_leace(X, z):
    from src.unlearning.concept_erasure import LeaceFitter
    print("Fitting LEACE...")
    z_oh = torch.nn.functional.one_hot(z.long(), num_classes=2).float()
    fitter = LeaceFitter.fit(X.double(), z_oh.double())
    eraser = fitter.eraser
    return eraser.P, eraser.bias


def fit_splince(X, z):
    from src.unlearning.splince.proj import proj
    print("Fitting SPLINCE...")
    X_np = X.double().numpy()
    z_np = z.reshape(-1, 1).numpy()
    y_np = np.zeros((len(z_np), 1))
    p = proj()
    p.fit(X_np, z_np, y_np, method='SPLINCE')
    return torch.tensor(p.P), torch.tensor(p.x_mean)


def verify_affine(X_pos, X_neg, P, b):
    """Check that the fitted map actually removes the linear concept signal."""
    P = P.double()
    b = b.double()
    mu_p = (X_pos.double() @ P.T + b).mean(0)
    mu_n = (X_neg.double() @ P.T + b).mean(0)
    before = (X_pos.double().mean(0) - X_neg.double().mean(0)).norm().item()
    after = (mu_p - mu_n).norm().item()
    print(f"  ||mean difference||: {before:.5f} -> {after:.5f} "
          f"({100 * after / max(before, 1e-12):.3f}% remaining)")
    return before, after


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--encoder_dir', required=True)
    parser.add_argument('--forget_organ', required=True,
                        help="organ to erase, e.g. PROSTATE / BREAST / LUNG / OVARIAN")
    parser.add_argument('--metadata', default=DEFAULT_METADATA)
    parser.add_argument('--output', required=True)
    parser.add_argument('--method', choices=['svd', 'leace', 'splince'], default='svd')
    parser.add_argument('--k', type=int, default=50, help="subspace size for svd")
    parser.add_argument('--discriminative', action='store_true',
                        help="svd: use target-vs-rest directions instead of target variance")
    parser.add_argument('--max_slides', type=int, default=300)
    parser.add_argument('--patches_per_slide', type=int, default=64)
    parser.add_argument('--center_on_target', action='store_true',
                        help="svd: centre on the target cohort mean rather than the pooled mean")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    if 'organ' not in df.columns:
        raise SystemExit(f"{args.metadata} has no 'organ' column")

    available = sorted(df['organ'].dropna().unique())
    if args.forget_organ not in available:
        raise SystemExit(
            f"organ '{args.forget_organ}' not in {args.metadata}.\n"
            f"Available: {available}\n"
            f"(note: data/metadata.csv historically wrote TCGA breast as 'BRCA' while "
            f"{DEFAULT_METADATA} uses 'BREAST' - prefer the latter)"
        )

    pos_df = df[df['organ'] == args.forget_organ]
    neg_df = df[df['organ'] != args.forget_organ]
    print(f"Fitting {args.method} eraser for {args.forget_organ} vs rest "
          f"({len(pos_df)} vs {len(neg_df)} slides)...")

    X_pos = load_patch_features(pos_df, args.encoder_dir, args.max_slides,
                                args.patches_per_slide, args.seed)
    X_neg = load_patch_features(neg_df, args.encoder_dir, args.max_slides,
                                args.patches_per_slide, args.seed + 1)
    print(f"X_pos: {list(X_pos.shape)}, X_neg: {list(X_neg.shape)}")

    if X_pos.size(0) < 10 or X_neg.size(0) < 10:
        raise SystemExit("Not enough readable features to fit an eraser.")

    parent = os.path.dirname(args.output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{args.output}.tmp.{os.getpid()}"

    if args.method == 'svd':
        U = fit_svd(X_pos, X_neg, args.k, args.discriminative)
        mu = X_pos.mean(dim=0) if args.center_on_target else \
            torch.cat([X_pos, X_neg], 0).mean(dim=0)
        P = torch.eye(U.shape[0], dtype=torch.float64) - U.double() @ U.double().T
        b = mu.double() - P @ mu.double()
        verify_affine(X_pos, X_neg, P, b)
        # Stored with the mean so it can be applied mean-preservingly; see
        # src/unlearning/apply.py.
        torch.save({'U': U, 'mu': mu}, tmp)
    else:
        X = torch.cat([X_pos, X_neg], dim=0)
        z = torch.cat([torch.ones(len(X_pos)), torch.zeros(len(X_neg))])
        P, mu = fit_leace(X, z) if args.method == 'leace' else fit_splince(X, z)
        weights = save_affine(tmp, P, mu)
        verify_affine(X_pos, X_neg, weights['P'].double(), weights['b'].double())

    os.replace(tmp, args.output)
    print(f"Saved {args.method} eraser to {args.output}")


if __name__ == "__main__":
    main()
