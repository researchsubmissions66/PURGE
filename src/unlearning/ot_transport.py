import torch

def matrix_sqrt(matrix, eps=1e-5):
    """
    Computes the square root and inverse square root of a symmetric positive semi-definite matrix.
    Uses eigenvalue decomposition.
    """
    # Ensure symmetry
    matrix = (matrix + matrix.T) / 2.0
    
    # Compute eigenvalues and eigenvectors
    L, V = torch.linalg.eigh(matrix)
    
    # Clamp negative eigenvalues to zero for numerical stability
    L = torch.clamp(L, min=eps)
    
    # Compute square root
    L_sqrt = torch.diag(torch.sqrt(L))
    sqrt_matrix = V @ L_sqrt @ V.T
    
    # Compute inverse square root
    L_inv_sqrt = torch.diag(1.0 / torch.sqrt(L))
    inv_sqrt_matrix = V @ L_inv_sqrt @ V.T
    
    return sqrt_matrix, inv_sqrt_matrix

def compute_coral_transformation(cov_source, cov_target, eps=1e-5):
    """
    Computes the CORAL transformation matrix W = cov_source^{-1/2} * cov_target^{1/2}
    """
    _, inv_sqrt_source = matrix_sqrt(cov_source, eps)
    sqrt_target, _ = matrix_sqrt(cov_target, eps)
    
    W = inv_sqrt_source @ sqrt_target
    return W

def apply_ot_unlearning(X, W, mu_source, mu_target):
    """
    Applies the Optimal Transport (CORAL) unlearning map.
    X_new = (X - mu_source) @ W + mu_target
    """
    X_centered = X - mu_source.unsqueeze(0)
    X_unlearned = X_centered @ W + mu_target.unsqueeze(0)
    return X_unlearned
