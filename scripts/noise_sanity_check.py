import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.unlearning.noise import apply_gaussian_noise, apply_dropout

def test_noise_baselines():
    print("=== Testing Naive Noise Baselines ===")
    # Dummy data: 1000 patches, 768 dimensions
    X = torch.randn(1000, 768)
    
    # 1. Gaussian Noise
    print("1. Gaussian Noise Test...")
    X_gaussian = apply_gaussian_noise(X, sigma=1.0)
    assert X_gaussian.shape == (1000, 768)
    # Check that noise was actually added
    assert not torch.allclose(X, X_gaussian)
    print("   [PASS] Isotropic Gaussian perturbation applied")

    # 2. Feature Dropout
    print("2. Feature Dropout Test...")
    X_dropout = apply_dropout(X, p=0.5)
    assert X_dropout.shape == (1000, 768)
    # Check that elements were zeroed out
    zeros = (X_dropout == 0).sum().item()
    total = X_dropout.numel()
    dropout_rate = zeros / total
    print(f"   [INFO] Empirical dropout rate: {dropout_rate:.3f} (Expected: ~0.500)")
    assert 0.45 < dropout_rate < 0.55
    print("   [PASS] Feature Dropout perturbation applied")

if __name__ == "__main__":
    try:
        test_noise_baselines()
        print("\nNoise Sanity Check PASSED. ✅")
    except Exception as e:
        import traceback
        print("\n❌ SANITY CHECK FAILED:")
        traceback.print_exc()
