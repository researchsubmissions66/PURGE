import warnings

import torch


def orthonormalize(M):
    """
    Orthonormalize the columns of M via QR, with a sign convention that makes the
    factorization unique (positive diagonal of R). Differentiable.

    M: [d, r] -> [d, r] with orthonormal columns.
    """
    Q, R = torch.linalg.qr(M)
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    sign = torch.sign(diag)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return Q * sign.unsqueeze(-2)


def svd_subspace(X, k=50):
    """
    Top-k principal directions of X.

    X: [N, D]
    Returns U_k [D, k] with orthonormal columns.

    NOTE: this is the *unsupervised* variance subspace of whatever X you pass in.
    Removing it deletes the directions the cohort varies most along, which are
    largely shared across cohorts - so it damages control tasks about as much as
    the target. For control-aware selection see spectral_subspace() in
    src/unlearning/spectral.py (lambda=0 reduces to this function).
    """
    if X.size(0) == 0:
        raise ValueError("Empty feature set provided for SVD.")

    k_eff = min(k, X.size(0) - 1, X.size(1))
    if k_eff < 1:
        raise ValueError(f"Not enough samples for SVD: got {X.size(0)} rows.")
    if k_eff < k:
        # Silently returning fewer directions would mislabel results: a run
        # reported as "k=256" on 111 samples is really k=110.
        warnings.warn(
            f"svd_subspace: requested k={k} but only {k_eff} directions are "
            f"estimable from {X.size(0)} samples of dim {X.size(1)}; using k={k_eff}.",
            RuntimeWarning, stacklevel=2,
        )
    k = k_eff

    X_centered = X - X.mean(dim=0, keepdim=True)
    # Right singular vectors are the principal components.
    _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    return Vh[:k].transpose(0, 1).contiguous()


def discriminative_subspace(X_pos, X_neg, k=50):
    """
    Directions along which the target cohort differs from the rest, rather than
    the directions the target cohort happens to vary along.

    Whitens by the pooled within-cohort covariance, then takes the top-k
    eigenvectors of

        delta delta^T + (C_pos - C_neg)^2

    which captures both the first-order (mean shift) and second-order (covariance
    shift) discrepancy between cohorts. The squared covariance difference is used
    so the term is PSD and ranks directions by the magnitude of the difference.

    Returns U_k [D, k] with orthonormal columns, in the ORIGINAL feature space, so
    it can be handed straight to remove_subspace().
    """
    if X_pos.size(0) < 2 or X_neg.size(0) < 2:
        raise ValueError("Both cohorts need at least 2 samples.")

    out_dtype = X_pos.dtype
    X_pos = X_pos.double()
    X_neg = X_neg.double()
    d = X_pos.size(1)
    k = min(k, d)

    mu_pos = X_pos.mean(dim=0)
    mu_neg = X_neg.mean(dim=0)
    C_pos = torch.cov(X_pos.t())
    C_neg = torch.cov(X_neg.t())

    n_pos, n_neg = X_pos.size(0), X_neg.size(0)
    Sw = (n_pos * C_pos + n_neg * C_neg) / (n_pos + n_neg)
    ridge = 1e-4 * torch.diagonal(Sw).mean().clamp_min(1e-12)
    Sw = Sw + ridge * torch.eye(d, dtype=torch.float64)

    L = torch.linalg.cholesky(Sw)

    def whiten_matrix(C):
        # L^-1 C L^-T
        tmp = torch.linalg.solve_triangular(L, C, upper=False)
        return torch.linalg.solve_triangular(L, tmp.t(), upper=False).t()

    delta_w = torch.linalg.solve_triangular(
        L, (mu_pos - mu_neg).unsqueeze(1), upper=False
    )
    cov_diff = whiten_matrix(C_pos) - whiten_matrix(C_neg)
    S = delta_w @ delta_w.t() + cov_diff @ cov_diff

    evals, evecs = torch.linalg.eigh(S)
    top = evecs[:, torch.argsort(evals, descending=True)[:k]]

    # A functional v in whitened coordinates acts on original x as (L^-T v)^T x.
    U = torch.linalg.solve_triangular(L.t(), top, upper=True)
    Q, _ = torch.linalg.qr(U)
    return Q.to(out_dtype)


def remove_direction(X, w, eps=1e-12):
    """Remove a single direction w [D] from X [N, D]."""
    w = w.to(X.device, X.dtype)
    denom = torch.dot(w, w).clamp_min(eps)
    coeff = (X @ w) / denom
    return X - coeff.unsqueeze(1) * w.unsqueeze(0)


def remove_subspace_affine(X, U, mu):
    """
    Mean-preserving null-space projection:

        z' = (z - mu) @ P.T + mu,   P = I - U U^T

    Prefer this over remove_subspace(). The discriminative signal lives in the
    CENTERED data, so subtracting the mean before projecting removes exactly the
    same information - but the plain form also deletes the embedding mean's
    component along U, and foundation-model embeddings put most of their norm in
    a few dominant directions. That inflates ||z - z'|| enormously for no gain in
    erasure, which breaks the "z' is close to z" requirement the attack needs to
    stay inconspicuous.
    """
    U = U.to(X.device, X.dtype)
    mu = mu.to(X.device, X.dtype)
    Xc = X - mu
    return Xc - (Xc @ U) @ U.T + mu


def remove_subspace(X, U):
    """
    Project X [N, D] onto the orthogonal complement of span(U), U [D, K]
    orthonormal. This is rank-deficient (rank D - K), so it genuinely destroys
    information in those K directions.
    """
    U = U.to(X.device, X.dtype)
    return X - (X @ U) @ U.T
