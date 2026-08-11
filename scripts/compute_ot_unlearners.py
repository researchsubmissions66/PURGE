import os
import glob
import h5py
import torch
import random
import argparse
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.unlearning.ot_transport import compute_coral_transformation

def load_random_patches(path_base, df, max_slides=500, max_patches_per_slide=100):
    valid_files = df['slide_id'].tolist()
    random.shuffle(valid_files)
    valid_files = valid_files[:max_slides]
    
    features = []
    count = 0
    for slide_id in valid_files:
        fpath = os.path.join(path_base, f"{slide_id}.h5")
        if not os.path.exists(fpath):
            continue
        try:
            with h5py.File(fpath, 'r') as f:
                feat = torch.tensor(f['features'][:])
                # Randomly subsample patches from slide
                if feat.shape[0] > max_patches_per_slide:
                    idx = torch.randperm(feat.shape[0])[:max_patches_per_slide]
                    feat = feat[idx]
                features.append(feat)
                count += 1
        except Exception:
            pass
            
    if not features:
        return torch.empty((0, 0))
    return torch.cat(features, dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', required=True)
    args = parser.parse_args()
    
    metadata = pd.read_csv("data/multi_benchmark_metadata.csv")
    
    # 1. Compute Global Distribution (Sample evenly from Prostate, Breast, Lung)
    print("Sampling features for Global distribution...")
    prostate_df = metadata[metadata['dataset'] == 'PANDA']
    breast_df = metadata[metadata['dataset'] == 'BRACS']
    lung_df = metadata[metadata['dataset'] == 'TCGA']
    
    X_prostate = load_random_patches(f"/work/hdd/bhwm/PANDA/20x_224px_0px_overlap/{args.encoder}", prostate_df)
    X_breast = load_random_patches(f"/work/hdd/bhwm/BRACS/20x_224px_0px_overlap/{args.encoder}", breast_df)
    X_lung = load_random_patches(f"/work/hdd/bhwm/trident_features/master_benchmark/20x_224px_0px_overlap/{args.encoder}", lung_df)
    
    X_global = torch.cat([X_prostate, X_breast, X_lung], dim=0)
    print(f"Global dataset shape: {X_global.shape}")
    
    mu_global = X_global.mean(dim=0)
    X_global_centered = X_global - mu_global
    cov_global = (X_global_centered.T @ X_global_centered) / (X_global.shape[0] - 1)
    
    # 2. Compute Organ-Specific Distributions and OT Maps
    os.makedirs("results/unlearners", exist_ok=True)
    
    organs = {
        'PROSTATE': X_prostate,
        'BREAST': X_breast,
        'LUNG': X_lung
    }
    
    for organ_name, X_org in organs.items():
        if X_org.shape[0] == 0:
            continue
            
        print(f"Computing OT Transformation for {organ_name} (Shape: {X_org.shape})")
        mu_org = X_org.mean(dim=0)
        X_org_centered = X_org - mu_org
        cov_org = (X_org_centered.T @ X_org_centered) / (X_org.shape[0] - 1)
        
        print(f"Computing CORAL Matrix...")
        W = compute_coral_transformation(cov_org, cov_global)
        
        out_file = f"results/unlearners/ot_{args.encoder}_{organ_name}.pt"
        torch.save({
            'W': W.cpu(),
            'mu_organ': mu_org.cpu(),
            'mu_global': mu_global.cpu()
        }, out_file)
        print(f"Saved OT unlearner to {out_file}")

if __name__ == "__main__":
    main()
