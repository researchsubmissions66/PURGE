import torch

def svd_subspace(X, k=50):
    """
    Computes the top K principal components (subspace) of the features X.
    X: [N, D] tensor
    Returns: U [D, K] orthonormal matrix
    """
    if X.size(0) == 0:
        raise ValueError("Empty feature set provided for SVD.")
        
    # Center the features
    X_centered = X - X.mean(dim=0, keepdim=True)
    
    # Perform SVD
    # X = U S V^T
    # We want V, the right singular vectors (which are the principal components)
    _, _, V = torch.svd(X_centered)
    
    # V is [D, D]. The top K components are the first K columns.
    U_k = V[:, :k]
    return U_k

def remove_direction(X, w, eps=1e-12):
    """
    Removes a single direction w from features X.
    X: [N, D] tensor
    w: [D] tensor (should be normalized)
    """
    w = w.to(X.device, X.dtype)
    denom = torch.dot(w, w).clamp_min(eps)
    coeff = (X @ w) / denom
    X_new = X - coeff.unsqueeze(1) * w.unsqueeze(0)
    return X_new

def remove_subspace(X, U):
    """
    Removes an orthonormal subspace U from X.
    X: [N, D] tensor
    U: [D, K] tensor with orthonormal columns
    """
    U = U.to(X.device, X.dtype)
    # Projection of X onto U is (X @ U)
    # Reconstructing that projection in D-dimensional space is (X @ U) @ U.T
    # Removing it: X - (X @ U) @ U.T
    return X - (X @ U) @ U.T
