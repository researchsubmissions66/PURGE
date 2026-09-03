import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.datasets.feature_dataset import select_dataset

N_SPLITS = 5
SPLIT_SEED = 42


def patient_folds(metadata_path, dataset_target, fold, n_splits=N_SPLITS, seed=SPLIT_SEED):
    """
    Patient-grouped K-fold split. Returns (train_patients, val_patients).

    Grouping is by patient_id, so all slides from one patient stay on one side.
    For PANDA/BACH/UBC-OCEAN patient_id is the slide id (one slide per case); for
    BRACS and TCGA it is a genuine patient with multiple slides.
    """
    df = select_dataset(pd.read_csv(metadata_path), dataset_target)
    if df.empty:
        raise ValueError(f"No rows in {metadata_path} for dataset '{dataset_target}'")

    patients = np.sort(df['patient_id'].unique())
    if len(patients) < n_splits:
        raise ValueError(
            f"{dataset_target} has {len(patients)} patients, need at least {n_splits}"
        )

    if not 0 <= fold < n_splits:
        raise ValueError(f"fold {fold} out of range for n_splits={n_splits}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = list(kf.split(patients))[fold]
    return patients[train_idx], patients[val_idx]


def inner_split(train_patients, val_frac=0.15, seed=SPLIT_SEED):
    """
    Carve a model-selection split out of the training patients.

    The outer fold's held-out patients are the TEST set and must not be used for
    early stopping or checkpoint selection - reporting max-over-epochs AUC on the
    same split you evaluate on is selection on the test set and biases the number
    upward. This inner split gives early stopping something legitimate to watch.
    """
    patients = np.asarray(train_patients)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(patients))
    n_val = max(1, int(round(val_frac * len(patients))))
    if len(patients) - n_val < 1:
        raise ValueError(f"Too few training patients ({len(patients)}) to carve a val split")
    val_idx, train_idx = order[:n_val], order[n_val:]
    return patients[train_idx], patients[val_idx]
