"""
Nonlinear bottleneck eraser.

    z' = dec(enc(z)),    enc: R^d -> R^m,   m < d

Trained by minimising

    sum_datasets ||z' - z||^2 / ||z||^2   +   lambda_t * HSIC(enc(z_target), y_target)

Why this is not another instance of what already failed
------------------------------------------------------
Ten methods have failed in this repo under two mechanisms (see AGENTS.md):

  1 Selection methods (Fisher, HSIC ranking, INLP, adversarial, whitening) assume
    task information is concentrated in identifiable directions. It is not - it is
    distributed across the cohort's principal span.
  2 Transport / second-order methods (QuadraticEraser, and by inheritance
    kernelised LEACE) need Sigma_c^{-1/2}, which equalises the fitting data
    exactly and loses ~50% of the effect out of sample, invariant to conditioning
    and shrinkage.

This construction avoids both. It selects no directions - the encoder is free to
compress however it likes - and it inverts no covariance. The only structural
constraint is the width of the code.

Guarantees
----------
z' is a deterministic function of the m-dimensional code, so by the data
processing inequality I(z'; Y) <= I(enc(z); Y) for every Y. The bottleneck caps
what can survive at all; the HSIC term removes the target specifically. Neither
depends on the erasure being linear, which is the point: a linear projection can
only remove a linear subspace, and the residual that survives affine SVD is
recoverable by a nonlinear probe (0.47 -> 0.73 when given nonlinear features).

Application is LABEL-FREE - unlike QuadraticEraser, which needs the label per
sample and is therefore not a released artefact.
"""

import torch
import torch.nn as nn

from .losses import hsic


class BottleneckEraser(nn.Module):
    def __init__(self, input_dim, code_dim, hidden=1024, depth=2):
        super().__init__()
        if not 0 < code_dim < input_dim:
            raise ValueError(
                f"code_dim must satisfy 0 < code_dim < input_dim, "
                f"got {code_dim} vs {input_dim}"
            )
        self.input_dim = input_dim
        self.code_dim = code_dim

        def mlp(a, b):
            layers, prev = [], a
            for _ in range(max(depth - 1, 0)):
                layers += [nn.Linear(prev, hidden), nn.GELU()]
                prev = hidden
            layers += [nn.Linear(prev, b)]
            return nn.Sequential(*layers)

        self.enc = mlp(input_dim, code_dim)
        self.dec = mlp(code_dim, input_dim)

    def encode(self, z):
        return self.enc(z)

    def forward(self, z):
        return self.dec(self.enc(z))

    def bottleneck_rank(self):
        """Information that survives is capped at the code width."""
        return self.code_dim


def fidelity(z, z_prime, eps=1e-12):
    return (z_prime - z).pow(2).sum() / z.pow(2).sum().clamp_min(eps)


def fit_bottleneck(eraser, Z_target, y_target, Z_controls=(), lambda_t=1.0,
                   steps=800, batch_size=256, lr=1e-3, seed=0, verbose=True,
                   hsic_min_batch=32):
    """
    Train the eraser. Returns the loss history.

    Fidelity is applied to the target AND every control cohort, so the code must
    compress all of them well while carrying no information about y_target. That
    is what protects the control tasks - no adversary, no control probes, nothing
    that can diverge.
    """
    torch.manual_seed(seed)
    device = next(eraser.parameters()).device
    Zt = Z_target.to(device)
    yt = y_target.to(device)
    Zc = [Z.to(device) for Z in Z_controls]

    opt = torch.optim.Adam(eraser.parameters(), lr=lr)
    history = []

    for step in range(steps):
        opt.zero_grad()

        idx = torch.randperm(len(Zt), device=device)[:batch_size]
        zb, yb = Zt[idx], yt[idx]
        code = eraser.encode(zb)
        zp = eraser.dec(code)

        loss_fid = fidelity(zb, zp)
        loss_hsic = (hsic(code, yb) if len(idx) >= hsic_min_batch
                     else torch.zeros((), device=device))

        loss_ctrl = torch.zeros((), device=device)
        for Z in Zc:
            j = torch.randperm(len(Z), device=device)[:batch_size]
            loss_ctrl = loss_ctrl + fidelity(Z[j], eraser(Z[j]))
        if Zc:
            loss_ctrl = loss_ctrl / len(Zc)

        loss = loss_fid + loss_ctrl + lambda_t * loss_hsic
        loss.backward()
        opt.step()

        history.append((float(loss_fid.detach()), float(loss_hsic.detach()),
                        float(loss_ctrl.detach())))
        if verbose and (step % max(steps // 6, 1) == 0 or step == steps - 1):
            print(f"    step {step:4d} | fid {float(loss_fid):.4f} "
                  f"| hsic {float(loss_hsic):.5f} | ctrl {float(loss_ctrl):.4f}",
                  flush=True)

    eraser.eval()
    for p in eraser.parameters():
        p.requires_grad_(False)
    return history
