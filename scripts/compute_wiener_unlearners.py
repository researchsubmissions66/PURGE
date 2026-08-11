import os
import h5py
import torch
import random
import argparse
import numpy as np
import pandas as pd

def load_random_patches(path_base, df, max_slides=500, max_patches_per_slide=100):
    valid_files = df['slide_id'].tolist()
    random.shuffle(valid_files)
    valid_files = valid_files[:max_slides]
    
    features = []
    for slide_id in valid_files:
        fpath = os.path.join(path_base, f"{slide_id}.h5")
        if not os.path.exists(fpath):
            continue
        try:
            with h5py.File(fpath, 'r') as f:
                feat = torch.tensor(f['features'][:])
                if feat.shape[0] > max_patches_per_slide:
                    idx = torch.randperm(feat.shape[0])[:max_patches_per_slide]
                    feat = feat[idx]
                features.append(feat)
        except Exception:
            pass
            
    if not features:
        return torch.empty((0, 0))
    return torch.cat(features, dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', required=True)
    parser.add_argument('--k_components', type=int, default=500)
    args = parser.parse_args()
    
    metadata = pd.read_csv("data/multi_benchmark_metadata.csv")
    os.makedirs("results/unlearners", exist_ok=True)
    
    organs_config = {
        'PROSTATE': ("PANDA", f"/work/hdd/bhwm/PANDA/20x_224px_0px_overlap/{args.encoder}"),
        'BREAST': ("BRACS", f"/work/hdd/bhwm/BRACS/20x_224px_0px_overlap/{args.encoder}")
    }
    
    for organ_name, (dataset_name, path_base) in organs_config.items():
        print(f"Sampling features for {organ_name}...")
        df = metadata[metadata['dataset'] == dataset_name]
        X = load_random_patches(path_base, df)
        
        if X.shape[0] == 0:
            continue
            
        print(f"Computing SVD/Eigenvalues for {organ_name} (Shape: {X.shape})")
        mu = X.mean(dim=0)
        X_centered = X - mu
        
        # Compute SVD: X = U S V^T
        U_full, S, V = torch.svd(X_centered)
        
        # Eigenvalues of Covariance Matrix: lambda_i = S_i^2 / (N - 1)
        lambdas = (S ** 2) / (X.shape[0] - 1)
        
        # Truncate to top K
        actual_k = min(args.k_components, V.shape[1])
        V_k = V[:, :actual_k]
        lambdas_k = lambdas[:actual_k]
        
        out_file = f"results/unlearners/wiener_{args.encoder}_{organ_name}_k{actual_k}.pt"
        torch.save({
            'V': V_k.cpu(),
            'lambdas': lambdas_k.cpu()
        }, out_file)
        print(f"Saved Wiener unlearner to {out_file}")

if __name__ == "__main__":
    main()
