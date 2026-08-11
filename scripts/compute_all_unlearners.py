import os
import h5py
import torch
import pandas as pd
from tqdm import tqdm
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.unlearning.subspace import svd_subspace

def get_random_features_from_lustre(df_subset, num_samples=200):
    if len(df_subset) > num_samples:
        df_subset = df_subset.sample(num_samples)
        
    features = []
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
        filename = str(row['filename'])
        if not filename.endswith('.h5'):
            filename = os.path.splitext(filename)[0] + '.h5'
        filepath = os.path.join(row['root_dir'], filename)
        try:
            with h5py.File(filepath, 'r') as f:
                slide_feats = torch.tensor(f['features'][:]).mean(dim=0)
                features.append(slide_feats)
        except Exception as e:
            pass
            
    if len(features) == 0:
        return torch.empty(0)
    return torch.stack(features)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', required=True)
    parser.add_argument('--organ', required=True)
    parser.add_argument('--k_components', type=int, default=50)
    args = parser.parse_args()
    
    metadata = pd.read_csv("data/multi_benchmark_metadata.csv")
    os.makedirs("results/unlearners", exist_ok=True)
    
    encoder = args.encoder
    organ = args.organ
    k_comp = args.k_components
    
    out_file = f"results/unlearners/{encoder}_{organ}_k{k_comp}.pt" if k_comp != 50 else f"results/unlearners/{encoder}_{organ}.pt"
    if os.path.exists(out_file):
        print(f"Skipping {out_file}, already exists.")
        return
        
    print(f"\nComputing SVD for {organ} on {encoder}...")
    pos_df = metadata[metadata['organ'] == organ].copy()
    pos_df['root_dir'] = pos_df['root_dir'] + "/" + encoder
    
    X_pos = get_random_features_from_lustre(pos_df, num_samples=300)
    print(f"Found {X_pos.shape[0]} valid slides.")
    
    if X_pos.size(0) > 10:
        actual_k = min(k_comp, X_pos.size(0) - 1, X_pos.size(1))
        U = svd_subspace(X_pos, k=actual_k)
        
        # Atomic save
        temp_out = f"{out_file}.tmp.{os.getpid()}"
        torch.save(U, temp_out)
        os.rename(temp_out, out_file)
        
        print(f"Saved to {out_file}")
    else:
        print("Not enough data.")

if __name__ == "__main__":
    main()
