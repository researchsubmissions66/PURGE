import os
import sys
import torch
import h5py
import pandas as pd
import argparse
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.unlearning.subspace import svd_subspace

def load_random_features(metadata_subset, encoder_dir, num_samples=1000):
    if len(metadata_subset) > num_samples:
        metadata_subset = metadata_subset.sample(num_samples)
    
    features = []
    for _, row in metadata_subset.iterrows():
        filename = str(row['filename']).split('.')[0] + '.h5'
        filepath = os.path.join(encoder_dir, filename)
        try:
            with h5py.File(filepath, 'r') as f:
                # We can mean-pool the bag of patches to get one vector per slide for computing the direction
                # or we can sample patches. Let's just mean-pool the slide for the direction
                slide_feats = torch.tensor(f['features'][:]).mean(dim=0)
                features.append(slide_feats)
        except Exception as e:
            pass
            
    if len(features) == 0:
        return torch.empty(0)
    return torch.stack(features)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_dir', required=True)
    parser.add_argument('--forget_organ', required=True)
    parser.add_argument('--metadata', default='data/metadata.csv')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    
    pos_df = df[df['organ'] == args.forget_organ]
    neg_df = df[df['organ'] != args.forget_organ]
    
    print(f"Fitting unlearner for {args.forget_organ} vs Rest...")
    X_pos = load_random_features(pos_df, args.encoder_dir, num_samples=200)
    X_neg = load_random_features(neg_df, args.encoder_dir, num_samples=200)
    
    print(f"X_pos shape: {X_pos.shape}, X_neg shape: {X_neg.shape}")
    
    if X_pos.size(0) == 0 or X_neg.size(0) == 0:
        print("Not enough data to compute direction.")
        return
        
    # We don't even need X_neg for simple PCA subspace erasure of the target concept
    U = svd_subspace(X_pos, k=50)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Use atomic save to prevent corruption if multiple jobs compute this at once
    temp_out = f"{args.output}.tmp.{os.getpid()}"
    torch.save(U, temp_out)
    os.rename(temp_out, args.output)
    
    print(f"Saved unlearning direction to {args.output}")

if __name__ == "__main__":
    main()
