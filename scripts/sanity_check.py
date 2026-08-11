import torch
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

print("=" * 80)
print("SANITY CHECK 1: Orthonormality of SVD subspace matrices")
print("=" * 80)

for enc in ['features_hoptimus0', 'features_gpfm', 'features_virchow2']:
    for organ in ['PROSTATE', 'BREAST', 'LUNG', 'OVARIAN']:
        U = torch.load(f'results/unlearners/{enc}_{organ}.pt')
        eye = U.T @ U
        off_diag_max = (eye - torch.eye(U.shape[1])).abs().max().item()
        print(f'{enc:25s} {organ:10s} shape={list(U.shape)}  off-diag-max={off_diag_max:.8f}')

print()
print("=" * 80)
print("SANITY CHECK 2: Cross-organ subspace overlap (Frobenius norm of U_a^T @ U_b)")
print("  If organs are distinct concepts, off-diagonal should be much less than diagonal")
print("=" * 80)

for enc in ['features_hoptimus0', 'features_gpfm', 'features_virchow2']:
    organs = ['PROSTATE', 'BREAST', 'LUNG', 'OVARIAN']
    Us = {}
    for organ in organs:
        Us[organ] = torch.load(f'results/unlearners/{enc}_{organ}.pt')
    
    print(f'\n{enc}:')
    header = f'{"":12s}'
    for o in organs:
        header += f'{o:12s}'
    print(header)
    for a in organs:
        row = f'{a:12s}'
        for b in organs:
            overlap = torch.linalg.norm(Us[a].T @ Us[b], 'fro').item()
            row += f'{overlap:12.4f}'
        print(row)

print()
print("=" * 80)
print("SANITY CHECK 3: Verify projection actually changes features")
print("=" * 80)

from src.unlearning.subspace import remove_subspace
import h5py, os

# Load a sample PANDA slide
enc = 'features_hoptimus0'
U_prostate = torch.load(f'results/unlearners/{enc}_PROSTATE.pt')
U_breast   = torch.load(f'results/unlearners/{enc}_BREAST.pt')

# Find one PANDA .h5 file
import pandas as pd
meta = pd.read_csv('data/multi_benchmark_metadata.csv')
panda_row = meta[meta['dataset'] == 'PANDA'].iloc[0]
fname = str(panda_row['filename'])
if not fname.endswith('.h5'):
    fname = os.path.splitext(fname)[0] + '.h5'
fpath = os.path.join(panda_row['root_dir'], enc, fname)

with h5py.File(fpath, 'r') as f:
    X = torch.tensor(f['features'][:])

X_no_prostate = remove_subspace(X, U_prostate)
X_no_breast   = remove_subspace(X, U_breast)

orig_norm = torch.linalg.norm(X, dim=1).mean().item()
prostate_removed_norm = torch.linalg.norm(X_no_prostate, dim=1).mean().item()
breast_removed_norm   = torch.linalg.norm(X_no_breast, dim=1).mean().item()

prostate_delta = torch.linalg.norm(X - X_no_prostate, dim=1).mean().item()
breast_delta   = torch.linalg.norm(X - X_no_breast, dim=1).mean().item()

print(f"Sample PANDA slide: {fname}")
print(f"  Original feature norm (mean):        {orig_norm:.4f}")
print(f"  After removing PROSTATE subspace:     {prostate_removed_norm:.4f}  (delta={prostate_delta:.4f})")
print(f"  After removing BREAST subspace:       {breast_removed_norm:.4f}  (delta={breast_delta:.4f})")
print()
print("  If PURGE works correctly, the PROSTATE delta should be LARGER than BREAST delta")
print(f"  on a Prostate slide, because PROSTATE subspace captures more variance of this slide.")
print(f"  Ratio: prostate_delta / breast_delta = {prostate_delta / breast_delta:.4f}")

print()
print("=" * 80)
print("SANITY CHECK 4: Raw JSON result consistency")
print("=" * 80)

import json, glob
files = sorted(glob.glob('results/*.json'))
print(f"Total result files: {len(files)}")
print(f"  Baseline files:   {len([f for f in files if 'baseline' in f])}")
print(f"  Unlearned files:  {len([f for f in files if 'unlearned' in f])}")

# Check all JSONs are valid and have expected keys
bad = 0
for f in files:
    try:
        d = json.load(open(f))
        assert 'best_val_auc' in d
        assert 'num_classes' in d
        assert 'fold' in d
    except:
        bad += 1
        print(f"  BAD FILE: {f}")
print(f"  Invalid JSON files: {bad}")
