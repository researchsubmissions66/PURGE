import os

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset


def select_dataset(df, dataset_target):
    """
    Filter a metadata frame to one benchmark. TCGA is split by organ because both
    cohorts live under dataset == 'TCGA'.
    """
    if dataset_target is None:
        return df
    if dataset_target == 'TCGA-BRCA':
        return df[(df['dataset'] == 'TCGA') & (df['organ'] == 'BREAST')]
    if dataset_target == 'TCGA-LUNG':
        return df[(df['dataset'] == 'TCGA') & (df['organ'] == 'LUNG')]
    return df[df['dataset'] == dataset_target]


def build_label_map(metadata_path, dataset_target):
    """
    Build the label -> index map ONCE, over the whole benchmark.

    This must not be derived per split: if a split happens to miss a class (a rare
    subtype, or the feature-file prefilter dropping one), a per-split map silently
    shifts the index of every later class and train/val stop agreeing on what a
    label means.
    """
    df = select_dataset(pd.read_csv(metadata_path), dataset_target)
    classes = sorted(df['label'].astype(str).unique())
    return {lbl: i for i, lbl in enumerate(classes)}


def to_h5_name(filename):
    filename = str(filename)
    return filename if filename.endswith('.h5') else os.path.splitext(filename)[0] + '.h5'


class FeatureDataset(Dataset):
    def __init__(self, metadata_path, encoder_dir, split_patients=None,
                 dataset_target=None, label_map=None):
        self.metadata = pd.read_csv(metadata_path)

        # The label map spans the whole benchmark, not just this split.
        self.label_map = label_map if label_map is not None else build_label_map(
            metadata_path, dataset_target
        )
        self.classes = sorted(self.label_map, key=self.label_map.get)
        self.num_classes = len(self.classes)

        self.metadata = select_dataset(self.metadata, dataset_target).reset_index(drop=True)

        if split_patients is not None:
            self.metadata = self.metadata[
                self.metadata['patient_id'].isin(split_patients)
            ].reset_index(drop=True)

        self.encoder_dir = encoder_dir

        # Drop slides whose features are not on this node's cache.
        present = self.metadata['filename'].map(
            lambda f: os.path.exists(os.path.join(encoder_dir, to_h5_name(f)))
        )
        n_before = len(self.metadata)
        self.metadata = self.metadata[present].reset_index(drop=True)

        n_missing = n_before - len(self.metadata)
        msg = f"Loaded {len(self.metadata)} valid slides from NVMe cache."
        if n_missing:
            msg += f" ({n_missing} of {n_before} missing from {encoder_dir})"
        print(msg)

        unknown = set(self.metadata['label'].astype(str)) - set(self.label_map)
        if unknown:
            raise ValueError(f"Labels absent from the label map: {sorted(unknown)}")

    def __len__(self):
        return len(self.metadata)

    def present_classes(self):
        """Class indices actually present in this split."""
        return sorted({self.label_map[str(l)] for l in self.metadata['label']})

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filepath = os.path.join(self.encoder_dir, to_h5_name(row['filename']))

        with h5py.File(filepath, 'r') as f:
            features = torch.tensor(f['features'][:])

        return features, self.label_map[str(row['label'])], row['patient_id']
