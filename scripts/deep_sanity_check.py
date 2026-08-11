import torch
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import Unlearning Methods
from src.unlearning.subspace import svd_subspace, remove_subspace
from src.unlearning.ot_transport import compute_coral_transformation, apply_ot_unlearning
from src.unlearning.wiener_filter import apply_wiener_filter

# Import MIL Models
from src.models.abmil import ABMIL
from src.models.meanmil import MeanMIL
from src.models.transmil import TransMIL

def test_unlearning_methods():
    print("=== Testing Unlearning Math ===")
    # Dummy data: 1000 patches, 768 dimensions
    X = torch.randn(1000, 768)
    
    # 1. SVD Subspace
    print("1. SVD Subspace Test...")
    U = svd_subspace(X, k=500)
    assert U.shape == (768, 500)
    
    # Test dynamic truncation to k=50
    U_k = U[:, :50]
    X_svd = remove_subspace(X, U_k)
    assert X_svd.shape == (1000, 768)
    print("   [PASS] SVD dynamic truncation & projection")

    # 2. Optimal Transport
    print("2. Optimal Transport Test...")
    cov_source = (X.T @ X) / 999
    cov_target = torch.eye(768) * 0.5  # Dummy target
    W = compute_coral_transformation(cov_source, cov_target)
    assert W.shape == (768, 768)
    mu = X.mean(dim=0)
    X_ot = apply_ot_unlearning(X, W, mu, torch.zeros(768))
    assert X_ot.shape == (1000, 768)
    print("   [PASS] OT Matrix Sqrt & Affine Transformation")

    # 3. Wiener Filter
    print("3. Spectral Wiener Filter Test...")
    U_full, S, V = torch.svd(X - mu)
    lambdas = (S ** 2) / 999
    X_wiener = apply_wiener_filter(X, V, lambdas, tau=0.1)
    assert X_wiener.shape == (1000, 768)
    print("   [PASS] Spectral Wiener Soft Projection")

def test_models():
    print("\n=== Testing MIL Architectures ===")
    X = torch.randn(100, 768) # 1 slide, 100 patches
    
    models = {
        'ABMIL': ABMIL(input_dim=768, num_classes=2),
        'MeanMIL': MeanMIL(input_dim=768, num_classes=2),
        'TransMIL': TransMIL(input_dim=768, num_classes=2)
    }
    
    for name, model in models.items():
        logits, A = model(X)
        assert logits.shape == (2,), f"{name} output shape incorrect: {logits.shape}"
        assert len(logits.shape) == 1, f"{name} logits should be 1D"
        print(f"   [PASS] {name} Forward Pass")

import traceback
if __name__ == "__main__":
    try:
        test_unlearning_methods()
        test_models()
        print("\nAll deep sanity checks PASSED. ✅")
    except Exception as e:
        print("\n❌ SANITY CHECK FAILED:")
        traceback.print_exc()
