"""
Erasure audits.

The central failure mode of representation-erasure attacks is applying an
*invertible* transform: the fixed probe used during training is fooled, but any
retrained probe recovers the concept exactly, because an invertible map is an
information-preserving bijection. These helpers make that failure loud instead
of silent.
"""

import torch


def linear_map(fn, input_dim, device='cpu', dtype=torch.float64, tol=1e-4):
    """
    Recover the matrix M with fn(z) == z @ M, if fn is linear.

    Returns (M, is_linear, bias) for fn(z) == z @ M + bias. An affine map counts
    as linear here: the bias is recovered as fn(0) and the rank of M is what bounds
    the information that survives.
    """
    eye = torch.eye(input_dim, device=device, dtype=dtype)
    with torch.no_grad():
        bias = fn(torch.zeros(1, input_dim, device=device, dtype=dtype))
        M = fn(eye) - bias
        double_out = fn(2.0 * eye) - bias

    scale = float(M.abs().max().clamp_min(1e-12))
    atol = tol * scale
    # Affine maps (LEACE, SPLINCE, affine SVD) are z -> z @ M + b. Recovering b
    # first and testing linearity of the remainder classifies them correctly
    # rather than dismissing them as "nonlinear", which would hide their rank.
    is_linear = bool(torch.allclose(double_out, 2.0 * M, atol=atol))
    return M, is_linear, bias


def audit_eraser(eraser, input_dim, device='cpu', tol=1e-6):
    """
    Inspect an eraser and report whether it can actually destroy information.

    Returns a dict with:
        linear          - whether the transform is linear
        rank            - numerical rank of the linear map (None if nonlinear)
        sigma_min/max   - extreme singular values (None if nonlinear)
        invertible      - True if the map provably preserves all information
        erased_dims     - number of annihilated dimensions
        note            - human-readable summary
    """
    was_training = eraser.training
    eraser.eval()

    # Probe in float64. Casting the input down to float32 costs ~1e-4 absolute
    # precision whenever the eraser does large-magnitude cancellation (an affine
    # eraser computes (z - mu) ... + mu), which swamps the rank threshold. We
    # cannot simply loosen that threshold instead: an invertible low-rank adapter
    # can legitimately have a condition number of ~1e4, so a loose cutoff would
    # report it as rank-deficient - the exact error this audit exists to catch.
    params = list(eraser.parameters())
    orig_dtype = params[0].dtype if params else None
    try:
        if orig_dtype is not None and orig_dtype.is_floating_point:
            eraser.double()
        M, is_linear, bias = linear_map(
            lambda z: eraser(z).double(), input_dim, device=device
        )
    finally:
        if orig_dtype is not None and orig_dtype.is_floating_point:
            eraser.to(orig_dtype)
        eraser.train(was_training)

    report = {
        'linear': is_linear,
        'rank': None,
        'sigma_min': None,
        'sigma_max': None,
        'invertible': None,
        'erased_dims': None,
        'affine_offset': None,
        'note': '',
    }

    if is_linear:
        report['affine_offset'] = float(bias.abs().max())
        s = torch.linalg.svdvals(M)
        rank = int((s > tol * s[0]).sum().item())
        report.update(
            rank=rank,
            sigma_min=float(s[-1]),
            sigma_max=float(s[0]),
            invertible=bool(s[-1] > tol * s[0]),
            erased_dims=input_dim - rank,
        )
        if report['invertible']:
            report['note'] = (
                f"INVERTIBLE (sigma_min={s[-1]:.3e}, cond={s[0] / s[-1]:.3e}). "
                "This transform erases NOTHING: a retrained probe recovers the "
                "target concept exactly."
            )
        else:
            report['note'] = (
                f"Rank-deficient: {input_dim - rank} of {input_dim} dimensions "
                "annihilated. Information in those directions is unrecoverable."
            )
    else:
        bottleneck = getattr(eraser, 'bottleneck_rank', lambda: None)()
        if bottleneck is not None:
            report.update(
                rank=bottleneck,
                invertible=False,
                erased_dims=input_dim - bottleneck,
                note=(
                    f"Nonlinear, but factors through a rank-{bottleneck} linear "
                    f"bottleneck; by the data processing inequality at most "
                    f"{bottleneck} of {input_dim} dimensions of information survive."
                ),
            )
        else:
            report['note'] = (
                "Nonlinear with no declared bottleneck; invertibility cannot be "
                "established analytically. Validate empirically with a retrained probe."
            )
    return report


def assert_erases(eraser, input_dim, device='cpu', tol=1e-6):
    """
    Raise if the eraser cannot destroy information. Call this before launching an
    expensive sweep.
    """
    report = audit_eraser(eraser, input_dim, device=device, tol=tol)
    if report['invertible'] is not False:
        raise ValueError(
            f"{type(eraser).__name__} does not erase information. {report['note']}"
        )
    return report
