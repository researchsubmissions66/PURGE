"""
Erasure as a spectral trace objective.

Affine SVD is already the solution to a loss, though it is never written down as
one. The subspace it removes is

    U* = argmax_{U^T U = I}  tr(U^T Sigma_t U)                              (1)

i.e. "take the subspace capturing the most variance of the target cohort". Written
this way the omission is obvious: (1) does not know the control tasks exist, which
is exactly where collateral damage comes from (erasing PANDA grading costs
detection 13 AUC points).

The generalisation is to price the controls into the same objective:

    L_lambda(U) = tr( U^T (Sigma_t - lambda * mean_c Sigma_c) U )           (2)

whose optimum is the top-k eigenvectors of the PENCIL Sigma_t - lambda*Sigma_c.
lambda = 0 recovers plain affine SVD exactly.

Why a difference and not a ratio
--------------------------------
The natural-looking objective is the Rayleigh ratio u^T Sigma_t u / u^T Sigma_c u,
i.e. background whitening. That was implemented and it FAILED completely (no
erasure at all): with n << d the control covariance is rank deficient, its inverse
explodes along the null space, and the top generalised eigenvectors are numerical
artifacts. Measured tell: mean variance ratio 2547.8.

The difference form needs no inverse and is always well defined. Same intent, no
singularity.

Why this is not the selection strategy that has already failed five times
-------------------------------------------------------------------------
Fisher / HSIC / iterative-INLP / adversarial all ranked directions by LABEL
DISCRIMINABILITY and all lost to plain variance ranking, because task information
is distributed across the cohort's principal span rather than concentrated in
discriminative directions. (2) is still a VARIANCE criterion over a span - it only
subtracts variance that belongs to the controls. It stays on the axis that works.

Differentiability
-----------------
tr(U^T (Sigma_t - lambda Sigma_c) U) is differentiable in the data, so the same
objective can be minimised through an encoder to make E_theta's own top-k cohort
subspace uninformative (plan section 19, internal poisoning) instead of bolting a
projection on afterwards. See spectral_erasure_loss().
"""

import torch


def _cov(X):
    Xc = X - X.mean(dim=0, keepdim=True)
    return (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)


def covariance_pencil(X_target, controls=(), lam=1.0, normalize=True):
    """
    Sigma_t - lam * mean_c Sigma_c.

    normalize: divide each covariance by its trace first, so `lam` means the same
    thing regardless of the cohorts' absolute scales.
    """
    St = _cov(X_target.double())
    if normalize:
        St = St / torch.diagonal(St).sum().clamp_min(1e-12)

    if not controls:
        return St

    Sc = torch.zeros_like(St)
    for X_c in controls:
        C = _cov(X_c.double())
        if normalize:
            C = C / torch.diagonal(C).sum().clamp_min(1e-12)
        Sc += C
    Sc /= len(controls)

    return St - lam * Sc


def spectral_subspace(X_target, k, controls=(), lam=1.0, normalize=True,
                      return_diagnostics=False):
    """
    Top-k eigenvectors of the covariance pencil: the closed-form maximiser of (2).

    lam = 0 is exactly plain affine SVD's subspace.

    Returns U [d, k] with orthonormal columns.
    """
    M = covariance_pencil(X_target, controls, lam=lam, normalize=normalize)
    M = 0.5 * (M + M.T)                                   # enforce symmetry
    evals, evecs = torch.linalg.eigh(M)
    order = torch.argsort(evals, descending=True)[:k]
    U = evecs[:, order].contiguous().float()

    if not return_diagnostics:
        return U

    St = _cov(X_target.double())
    var_captured = float(torch.diagonal(U.double().T @ St @ U.double()).sum()
                         / torch.diagonal(St).sum().clamp_min(1e-12))
    diag = {
        'eigenvalues': evals[order].tolist(),
        'n_negative_eigenvalues': int((evals[order] < 0).sum()),
        'target_variance_captured': var_captured,
        'lam': lam,
    }
    if controls:
        for i, X_c in enumerate(controls):
            Sc = _cov(X_c.double())
            diag[f'control{i}_variance_captured'] = float(
                torch.diagonal(U.double().T @ Sc @ U.double()).sum()
                / torch.diagonal(Sc).sum().clamp_min(1e-12))
    return U, diag


def spectral_erasure_loss(Z_target, U, Z_controls=(), lam=1.0, normalize=True):
    """
    Differentiable form of (2), for training an encoder rather than fitting a
    projection.

    MINIMISING this w.r.t. encoder parameters pushes the target cohort's variance
    OUT of span(U) while leaving the controls' variance there, i.e. it makes the
    encoder's own dominant subspace uninformative about the target.

    Z_target : [n, d] embeddings of the target cohort (requires grad)
    U        : [d, k] the subspace held fixed for this step
    """
    U = U.to(Z_target.dtype)

    Zc = Z_target - Z_target.mean(dim=0, keepdim=True)
    proj = Zc @ U
    num = proj.pow(2).sum()
    if normalize:
        num = num / Zc.pow(2).sum().clamp_min(1e-12)
    loss = num

    if Z_controls:
        ctrl = 0.0
        for Z_c in Z_controls:
            Cc = Z_c - Z_c.mean(dim=0, keepdim=True)
            t = (Cc @ U).pow(2).sum()
            if normalize:
                t = t / Cc.pow(2).sum().clamp_min(1e-12)
            ctrl = ctrl + t
        loss = loss - lam * ctrl / len(Z_controls)

    return loss
