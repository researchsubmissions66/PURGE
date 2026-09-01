"""
Fast linear-probe check of an eraser on mean-pooled slide features.

This is a cheap lower bound on recoverability, not a substitute for the MIL
protocol: a linear probe failing on z' does not mean a nonlinear one will.
Read it as "did the affine eraser at least remove the linearly decodable signal".
"""

import argparse
import concurrent.futures
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.datasets.feature_dataset import select_dataset, to_h5_name
from src.evaluation.metrics import macro_ovr_auc
from src.unlearning.apply import apply_affine


def load_mean_features(df, encoder_dir, workers=32):
    def load(row):
        try:
            with h5py.File(os.path.join(encoder_dir, to_h5_name(row['filename'])), 'r') as f:
                return torch.tensor(f['features'][:]).mean(dim=0).numpy()
        except (OSError, KeyError):
            return None

    rows = [r for _, r in df.iterrows()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(tqdm(pool.map(load, rows), total=len(rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--encoder_dir', required=True)
    parser.add_argument('--dataset', default='PANDA')
    parser.add_argument('--metadata', default='data/multi_benchmark_metadata.csv')
    parser.add_argument('--unlearner', required=True,
                        help="affine eraser (.pt with keys P and b)")
    parser.add_argument('--max_slides', type=int, default=1000)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    df = select_dataset(pd.read_csv(args.metadata), args.dataset)
    if len(df) > args.max_slides:
        df = df.sample(args.max_slides, random_state=args.seed)
    df = df.reset_index(drop=True)

    print(f"Loading mean-pooled features for {len(df)} {args.dataset} slides...")
    loaded = load_mean_features(df, args.encoder_dir)

    features, labels = [], []
    for vec, label in zip(loaded, df['label']):
        if vec is not None:
            features.append(vec)
            labels.append(str(label))

    if not features:
        raise SystemExit(f"No readable features under {args.encoder_dir}")

    X = np.stack(features)
    classes = sorted(set(labels))
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[l] for l in labels])
    num_classes = len(classes)
    print(f"Loaded {X.shape[0]} slides, {num_classes} classes: {classes}")

    weights = torch.load(args.unlearner, map_location='cpu', weights_only=True)
    if not (isinstance(weights, dict) and 'P' in weights and 'b' in weights):
        raise SystemExit(
            f"{args.unlearner} is not an affine eraser. Expected keys 'P' and 'b'; "
            "fit one with scripts/fit_unlearner.py --method leace|splince."
        )
    X_erased = apply_affine(torch.tensor(X, dtype=torch.float32), weights).numpy()

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    baseline_aucs, erased_aucs = [], []

    print("\nRunning logistic-regression probes...")
    for train_idx, test_idx in skf.split(X, y):
        for source, bucket in ((X, baseline_aucs), (X_erased, erased_aucs)):
            clf = LogisticRegression(max_iter=1000, class_weight='balanced')
            clf.fit(source[train_idx], y[train_idx])
            probs = clf.predict_proba(source[test_idx])
            if num_classes == 2:
                probs = probs[:, 1]
            auc, _ = macro_ovr_auc(y[test_idx], probs, num_classes, strict=False)
            if auc is not None:
                bucket.append(auc)

    print("\n" + "=" * 55)
    print(f"Dataset:  {args.dataset}")
    print(f"Eraser:   {args.unlearner}")
    print(f"Baseline AUC: {np.mean(baseline_aucs):.4f} +/- {np.std(baseline_aucs):.4f}")
    print(f"Erased AUC:   {np.mean(erased_aucs):.4f} +/- {np.std(erased_aucs):.4f}")
    print("Chance for macro one-vs-rest AUC is 0.50.")
    print("=" * 55)


if __name__ == "__main__":
    main()
