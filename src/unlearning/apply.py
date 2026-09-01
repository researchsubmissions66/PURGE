"""
One place that knows how to build, save, load and apply every eraser.

SVD erasers
-----------
Stored as {'U': [d, k], 'mu': [d]} and applied mean-preservingly:

    z' = (z - mu) - ((z - mu) @ U) @ U.T + mu

This is the DEFAULT and what `--unlearn_method svd` does. The plain form
z @ (I - U U^T) erases identically (the discriminative signal lives in the
centred data) but additionally deletes the embedding mean's component along U.
Foundation-model embeddings concentrate most of their norm in a few dominant
directions, so that inflates ||z - z'|| enormously for zero gain: measured
cos(z, z') 0.43 -> 0.82 and 0.59 -> 0.86 on BACH and TCGA-LUNG at identical AUC.
Pass affine=False only to reproduce the plain form as an ablation.

Affine convention
-----------------
Linear/affine erasers (svd, leace, splince) are stored as {'P': [d, d], 'b': [d]}
and applied as

    x' = x @ P.T + b

with b already absorbing the mean correction, i.e. b = mu - P @ mu for a
mean-centred fit. Writing `x @ P.T + mu` instead - adding the raw mean rather
than the part of it the projection removed - leaves a residual offset of mu @ P.T
and corrupts the erasure. save_affine() enforces the convention.
"""

import os
import warnings

import torch

from .low_rank import LowRankEraser
from .noise import apply_dropout, apply_gaussian_noise
from .subspace import remove_subspace, remove_subspace_affine

AFFINE_METHODS = ('leace', 'splince')
STOCHASTIC_METHODS = ('gaussian', 'dropout')
METHODS = ('none', 'svd', 'low_rank') + AFFINE_METHODS + STOCHASTIC_METHODS


def save_affine(path, P, mu, validate=True):
    """
    Store an affine eraser in the canonical convention.

    P:  [d, d] projection matrix fitted on mean-centred data
    mu: [d] training mean
    """
    P = P.detach().cpu().double()
    mu = mu.detach().cpu().double()
    b = mu - P @ mu

    if validate:
        # The fixed point of the affine map must be the mean itself.
        recon = mu @ P.T + b
        err = (recon - mu).abs().max().item()
        if err > 1e-6 * max(mu.abs().max().item(), 1.0):
            raise ValueError(
                f"affine eraser fails its fixed-point check (max err {err:.3e}); "
                "P and mu are inconsistent"
            )

    torch.save({'P': P.float(), 'b': b.float()}, path)
    return {'P': P.float(), 'b': b.float()}


def apply_affine(X, weights):
    P = weights['P'].to(X.device, X.dtype)
    b = weights['b'].to(X.device, X.dtype)
    return X @ P.T + b


def build_eraser(method, path=None, device='cpu', input_dim=None, k=None,
                 sigma=1.0, dropout_p=0.5, rank=None, affine=True):
    """
    Return (fn, handle) where fn maps a [N, D] feature bag to its erased version
    and handle is the underlying object (module or weight dict), or (None, None)
    when method == 'none'.

    affine: for 'svd', use the mean-preserving form (default). See module docstring.
    """
    if method in ('none', None):
        return None, None

    if method == 'gaussian':
        return (lambda X: apply_gaussian_noise(X, sigma=sigma)), {'sigma': sigma}

    if method == 'dropout':
        return (lambda X: apply_dropout(X, p=dropout_p)), {'dropout_p': dropout_p}

    if path is None:
        raise ValueError(f"method '{method}' requires a fitted eraser file")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Eraser file {path} does not exist!")

    if method == 'low_rank':
        if input_dim is None:
            raise ValueError("low_rank eraser needs input_dim")
        module = LowRankEraser(input_dim=input_dim, rank=rank or 16).to(device)
        module.load_state_dict(torch.load(path, map_location=device))
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
        return module, module

    weights = torch.load(path, map_location=device)

    if method in AFFINE_METHODS:
        if not (isinstance(weights, dict) and 'P' in weights and 'b' in weights):
            raise ValueError(
                f"{method} eraser at {path} must be a dict with keys 'P' and 'b'"
            )
        weights = {k_: v.to(device) for k_, v in weights.items()}
        return (lambda X: apply_affine(X, weights)), weights

    if method == 'svd':
        mu = None
        if isinstance(weights, dict):
            if 'U' not in weights:
                raise ValueError(
                    f"svd eraser at {path} must hold 'U' (and ideally 'mu'); "
                    f"got keys {sorted(weights)}"
                )
            U = weights['U'].to(device)
            mu = weights.get('mu')
            if mu is not None:
                mu = mu.to(device)
        else:
            # Legacy checkpoint: bare [d, k] basis with no stored mean.
            U = weights.to(device)

        if k is not None:
            U = U[:, :min(k, U.shape[1])]

        if mu is None or not affine:
            if affine and mu is None:
                warnings.warn(
                    f"{path} has no stored mean; falling back to the plain "
                    "projection, which distorts embeddings far more for the same "
                    "erasure. Refit with scripts/fit_unlearner.py.",
                    RuntimeWarning,
                )
            return (lambda X: remove_subspace(X, U)), U
        return (lambda X: remove_subspace_affine(X, U, mu)), {'U': U, 'mu': mu}

    raise ValueError(f"Unknown unlearn method '{method}'. Expected one of {METHODS}.")
