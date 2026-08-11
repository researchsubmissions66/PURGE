import os
import h5py
import torch
import pandas as pd
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(self, metadata_path, encoder_dir, split_patients=None, dataset_target=None):
        self.metadata = pd.read_csv(metadata_path)
        
        # Filter by dataset first (e.g. PANDA, BACH)
        if dataset_target is not None:
            self.metadata = self.metadata[self.metadata['dataset'] == dataset_target].reset_index(drop=True)
            
        if split_patients is not None:
            self.metadata = self.metadata[self.metadata['patient_id'].isin(split_patients)].reset_index(drop=True)
            
        self.encoder_dir = encoder_dir
        
        # Pre-filter metadata to ensure all files exist in the NVMe cache
        valid_indices = []
        for idx, row in self.metadata.iterrows():
            filename = str(row['filename'])
            if not filename.endswith('.h5'):
                filename = os.path.splitext(filename)[0] + '.h5'
            if os.path.exists(os.path.join(self.encoder_dir, filename)):
                valid_indices.append(idx)
        self.metadata = self.metadata.loc[valid_indices].reset_index(drop=True)
        print(f"Loaded {len(self.metadata)} valid slides from NVMe cache.")
        
        # Dynamically build labels
        self.classes = sorted(self.metadata['label'].astype(str).unique())
        self.label_map = {lbl: i for i, lbl in enumerate(self.classes)}
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Convert filename to .h5 format (in these datasets, it's just stem + .h5)
        # e.g. e0ca5e9b18d9d563d7b6a4fe3d919f89.tiff -> e0ca5e9b18d9d563d7b6a4fe3d919f89.h5
        filename = str(row['filename'])
        if not filename.endswith('.h5'):
            filename = os.path.splitext(filename)[0] + '.h5'
        filepath = os.path.join(self.encoder_dir, filename)
        
        with h5py.File(filepath, 'r') as f:
            features = torch.tensor(f['features'][:])
            
        label = self.label_map[str(row['label'])]
            
        return features, label, row['patient_id']
