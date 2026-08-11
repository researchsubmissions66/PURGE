import torch

def apply_wiener_filter(X, U, lambdas, tau=0.1):
    """
    Applies the Spectral Wiener Filter for soft subspace projection.
    
    Args:
        X: Tensor of shape [N, D] (the features to unlearn)
        U: Tensor of shape [D, K] (the top K eigenvectors of the concept)
        lambdas: Tensor of shape [K] (the top K eigenvalues of the concept)
        tau: Float, the regularization parameter for the Wiener filter.
             Lower tau = closer to hard SVD projection.
             Higher tau = softer projection, leaving more variance intact.
             
    Returns:
        X_new: Tensor of shape [N, D] (the unlearned features)
    """
    # Compute Wiener weights for each component: w_i = lambda_i / (lambda_i + tau)
    # If lambda_i is huge, w_i ~ 1.0 (component is heavily suppressed)
    # If lambda_i is small, w_i ~ 0.0 (component is preserved)
    weights = lambdas / (lambdas + tau)
    
    # Construct the diagonal weight matrix W
    W = torch.diag(weights).to(X.device)
    U = U.to(X.device)
    
    # The projection matrix P = I - U W U^T
    # Applied to X: X_new = X - X @ (U W U^T)
    # Which evaluates faster as: X_new = X - ((X @ U) @ W) @ U^T
    
    X_proj = X @ U             # [N, K]
    X_weighted = X_proj @ W    # [N, K]
    X_reconstructed = X_weighted @ U.T # [N, D]
    
    X_new = X - X_reconstructed
    return X_new
