import torch
import torch.nn.functional as F

MAX_GEOMETRY_PATCHES = 4000


def cosine_preservation_loss(z, z_prime):
    """
    1 - mean cos(z, z'). Keeps each transformed embedding pointing the same way as
    the original (plan section 8).
    """
    sim = F.cosine_similarity(z, z_prime, dim=-1)
    return 1.0 - sim.mean()


def euclidean_preservation_loss(z, z_prime):
    """Mean ||z - z'||^2 per embedding, normalized by ||z||^2 so it is scale-free."""
    num = (z - z_prime).pow(2).sum(dim=-1)
    den = z.pow(2).sum(dim=-1).clamp_min(torch.finfo(z.dtype).tiny)
    return (num / den).mean()


def pairwise_geometry_loss(z, z_prime, max_patches=MAX_GEOMETRY_PATCHES):
    """
    ||S - S'||_F^2 (as an MSE) where S is the pairwise *cosine* similarity matrix
    (plan section 9). Preserves the relative geometry of the patch cloud.

    Accepts [N, D] or [B, N, D]. Patches are subsampled to bound the N^2 cost.
    """
    if z.dim() == 2 and z.shape[0] > max_patches:
        idx = torch.randperm(z.shape[0], device=z.device)[:max_patches]
        z, z_prime = z[idx], z_prime[idx]
    elif z.dim() == 3 and z.shape[1] > max_patches:
        idx = torch.randperm(z.shape[1], device=z.device)[:max_patches]
        z, z_prime = z[:, idx, :], z_prime[:, idx, :]

    z_norm = F.normalize(z, p=2, dim=-1)
    z_prime_norm = F.normalize(z_prime, p=2, dim=-1)

    if z.dim() == 2:
        S = z_norm @ z_norm.t()
        S_prime = z_prime_norm @ z_prime_norm.t()
    elif z.dim() == 3:
        S = torch.bmm(z_norm, z_norm.transpose(1, 2))
        S_prime = torch.bmm(z_prime_norm, z_prime_norm.transpose(1, 2))
    else:
        raise ValueError(f"Unsupported dimensions for pairwise_geometry_loss: {z.dim()}")

    return F.mse_loss(S_prime, S)


def uniform_target_loss(logits):
    """
    Adversarial objective for the eraser: push the probe's posterior towards
    uniform. Equals the cross-entropy from a uniform target up to a constant, and
    is bounded below (unlike -CE, which diverges).

    logits: [C] or [B, C].
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return -log_probs.mean()


def _rbf_kernel(X, sigma=None):
    """RBF Gram matrix with the median heuristic bandwidth."""
    sq_dists = torch.cdist(X, X).pow(2)
    if sigma is None:
        med = sq_dists.detach().flatten().median().clamp_min(1e-12)
        sigma_sq = med / 2.0
    else:
        sigma_sq = sigma ** 2
    return torch.exp(-sq_dists / (2.0 * sigma_sq))


def _delta_kernel(y):
    """Categorical kernel: 1 where labels agree, 0 otherwise."""
    y = y.view(-1, 1)
    return (y == y.t()).to(torch.get_default_dtype())


def hsic(X, y, sigma=None):
    """
    Biased HSIC estimator between representations X [N, D] and categorical labels
    y [N] (plan section 11).

    This is a classifier-free dependence measure: driving it to zero removes the
    statistical dependence between representation and target rather than merely
    defeating one probe, so it cannot be gamed by a probe that has not converged.

    Needs N >= 4 to be meaningful.
    """
    n = X.shape[0]
    if n < 4:
        return X.sum() * 0.0

    K = _rbf_kernel(X, sigma)
    L = _delta_kernel(y).to(X.device, X.dtype)

    H = torch.eye(n, device=X.device, dtype=X.dtype) - 1.0 / n
    KH = K @ H
    LH = L @ H
    return (KH * LH.t()).sum() / ((n - 1) ** 2)
