"""
The tests that matter: an eraser must actually destroy information.

test_low_rank_adapter_is_invertible pins the negative result that motivated the
redesign - a residual adapter cannot erase, so it must never be used for a claim.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.unlearning.apply import apply_affine, save_affine
from src.unlearning.audit import assert_erases, audit_eraser
from src.unlearning.losses import cosine_preservation_loss, hsic, pairwise_geometry_loss
from src.unlearning.low_rank import LowRankEraser
from src.unlearning.subspace import orthonormalize
from src.unlearning.subspace import (discriminative_subspace, orthonormalize,
                                     remove_subspace, remove_subspace_affine,
                                     svd_subspace)

D, R = 64, 8


def linear_r2(X, y):
    """
    R^2 of the best linear readout of y from X.

    Uses the pseudo-inverse deliberately: after a projection X is rank-deficient,
    and torch.linalg.lstsq's default 'gels' driver assumes full rank and returns
    garbage on such input (observed: R^2 ~ 0 where the true value was ~0.9).
    """
    Xb = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], dim=1).double()
    yy = y.double().unsqueeze(1)
    w = torch.linalg.pinv(Xb) @ yy
    resid = (Xb @ w).squeeze(1) - yy.squeeze(1)
    return float((1 - resid.var() / yy.squeeze(1).var()).detach())



def test_orthonormalize_produces_orthonormal_columns():
    Q = orthonormalize(torch.randn(D, R))
    assert torch.allclose(Q.T @ Q, torch.eye(R), atol=1e-5)


def test_low_rank_adapter_is_invertible():
    """
    Negative control. z' = (I + alpha*B*A)z is a bijection, so it erases nothing
    however it is trained. Kept as a test so this cannot silently come back.
    """
    eraser = LowRankEraser(input_dim=D, rank=R)
    with torch.no_grad():
        eraser.B.weight.normal_(0, 0.1)   # move off the zero init
    report = audit_eraser(eraser, D)
    assert report['linear']
    assert report['invertible'] is True
    assert report['rank'] == D

    with pytest.raises(ValueError, match="does not erase"):
        assert_erases(eraser, D)


def test_svd_subspace_is_orthonormal_and_clamped():
    X = torch.randn(20, D)
    U = svd_subspace(X, k=50)          # k exceeds the sample count
    assert U.shape[1] <= 19
    assert torch.allclose(U.T @ U, torch.eye(U.shape[1]), atol=1e-4)


def test_discriminative_subspace_beats_variance_subspace():
    """
    The discriminative direction must separate the cohorts better than the target
    cohort's own top variance direction.

    Separation is the Fisher score |u.(mu_p - mu_n)| / sqrt(u^T Sw u). Note this
    is NOT the raw mean-difference direction: with correlated within-cohort noise
    the optimal separating direction is Sw^-1 delta, which points elsewhere.
    """
    torch.manual_seed(0)
    shift = torch.zeros(D)
    shift[3] = 3.0

    noise = torch.randn(D, D) * 0.4
    cov = noise @ noise.T + torch.eye(D)
    L = torch.linalg.cholesky(cov)

    X_neg = torch.randn(600, D) @ L.T
    X_pos = torch.randn(600, D) @ L.T + shift

    delta = X_pos.mean(0) - X_neg.mean(0)
    Sw = 0.5 * (torch.cov(X_pos.T) + torch.cov(X_neg.T))

    def fisher(u):
        u = u / u.norm()
        return (u @ delta).abs() / (u @ Sw @ u).sqrt()

    u_disc = discriminative_subspace(X_pos, X_neg, k=1)[:, 0]
    u_var = svd_subspace(X_pos, k=1)[:, 0]

    assert fisher(u_disc) > fisher(u_var), (
        f"discriminative {fisher(u_disc):.3f} did not beat variance {fisher(u_var):.3f}"
    )


def test_discriminative_subspace_is_orthonormal():
    torch.manual_seed(0)
    X_pos = torch.randn(200, D) + 1.0
    X_neg = torch.randn(200, D)
    U = discriminative_subspace(X_pos, X_neg, k=5)
    assert U.shape == (D, 5)
    assert torch.allclose(U.T @ U, torch.eye(5), atol=1e-4)


def test_affine_convention_round_trips_the_mean():
    torch.manual_seed(0)
    U = orthonormalize(torch.randn(D, R))
    P = torch.eye(D) - U @ U.T
    mu = torch.randn(D) * 3.0

    path = '/tmp/_purge_affine_test.pt'
    weights = save_affine(path, P, mu)
    os.remove(path)

    X = torch.randn(64, D) + mu
    out = apply_affine(X, weights)

    # Correct convention: x' = (x - mu) @ P.T + mu, so the training mean is a
    # fixed point of the affine map.
    expected = (X - mu) @ P.T + mu
    assert torch.allclose(out, expected, atol=1e-4)
    # The naive (buggy) form x @ P.T + mu differs by exactly mu @ P.T.
    naive = X @ P.T + mu
    assert not torch.allclose(naive, expected, atol=1e-3)


def test_save_affine_stores_the_mean_correction():
    torch.manual_seed(0)
    U = orthonormalize(torch.randn(D, R))
    P = torch.eye(D) - U @ U.T
    mu = torch.randn(D) * 3.0

    path = '/tmp/_purge_affine_test2.pt'
    weights = save_affine(path, P, mu)
    os.remove(path)

    # b must be mu - P @ mu, not the raw mean.
    assert torch.allclose(weights['b'].double(), (mu - P @ mu).double(), atol=1e-5)
    assert not torch.allclose(weights['b'].double(), mu.double(), atol=1e-3)
    # The mean is a fixed point of the stored map.
    recovered = mu.double() @ weights['P'].double().T + weights['b'].double()
    assert torch.allclose(recovered, mu.double(), atol=1e-4)


def test_preservation_losses_are_zero_at_identity():
    z = torch.randn(50, D)
    assert cosine_preservation_loss(z, z).abs() < 1e-6
    assert pairwise_geometry_loss(z, z).abs() < 1e-6


def test_hsic_detects_and_clears_dependence():
    torch.manual_seed(0)
    y = torch.randint(0, 2, (128,))
    dependent = torch.randn(128, 4) + y.float().unsqueeze(1) * 5.0
    independent = torch.randn(128, 4)
    assert hsic(dependent, y) > hsic(independent, y) * 5


def test_affine_removal_erases_identically_but_distorts_less():
    """
    The mean-preserving form is the default because it is strictly better: the
    discriminative signal lives in the centred data, so both forms delete exactly
    the same information, but the plain form additionally deletes the mean's
    component along U - and embeddings concentrate most of their norm there.
    """
    torch.manual_seed(0)
    mu = torch.randn(D) * 20.0                 # large mean, as in real embeddings
    Xc = torch.randn(200, D)
    X = Xc + mu

    U = svd_subspace(X, k=R)
    plain = remove_subspace(X, U)
    affine = remove_subspace_affine(X, U, X.mean(0))

    # Identical up to a constant offset => identical information content.
    delta = affine - plain
    assert delta.std(dim=0).max() < 1e-3, "the two forms must differ only by a constant"

    # The plain form sends the erased subspace to zero; the affine form sends it
    # to the CONSTANT mu @ U. Both destroy the same information - a constant is
    # identical across samples and so carries none - but only the plain form is
    # literally zero there. Assert the invariant that actually matters.
    assert (plain @ U).abs().max() < 1e-3
    proj = affine @ U
    assert proj.std(dim=0).max() < 1e-3, "erased directions must be constant"
    assert proj.abs().max() > 1e-3, "and for a large mean, that constant is nonzero"

    # But the affine form stays far closer to the original embedding.
    cos = torch.nn.functional.cosine_similarity(X, affine, dim=-1).mean()
    cos_plain = torch.nn.functional.cosine_similarity(X, plain, dim=-1).mean()
    assert cos > cos_plain, f"affine {cos:.4f} should beat plain {cos_plain:.4f}"


def test_build_eraser_defaults_to_affine_svd():
    import tempfile

    from src.unlearning.apply import build_eraser

    torch.manual_seed(0)
    X = torch.randn(100, D) + torch.randn(D) * 20.0
    U = svd_subspace(X, k=R)
    mu = X.mean(0)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    torch.save({'U': U, 'mu': mu}, path)
    try:
        fn_affine, _ = build_eraser('svd', path=path, input_dim=D)
        fn_plain, _ = build_eraser('svd', path=path, input_dim=D, affine=False)
        assert torch.allclose(fn_affine(X), remove_subspace_affine(X, U, mu), atol=1e-4)
        assert torch.allclose(fn_plain(X), remove_subspace(X, U), atol=1e-4)
        # The default must be the affine one.
        assert not torch.allclose(fn_affine(X), fn_plain(X), atol=1e-2)
    finally:
        os.remove(path)


def test_legacy_bare_tensor_svd_warns():
    import tempfile
    import warnings as _w

    from src.unlearning.apply import build_eraser

    U = svd_subspace(torch.randn(100, D), k=R)
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    torch.save(U, path)
    try:
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            build_eraser('svd', path=path, input_dim=D)
        assert any("no stored mean" in str(c.message) for c in caught)
    finally:
        os.remove(path)


def test_audit_classifies_affine_erasers_correctly():
    """
    LEACE, SPLINCE and affine SVD are z -> z @ P.T + b. Before the bias was
    recovered separately, audit_eraser saw fn(0) != 0, declared them nonlinear,
    and reported "invertibility cannot be established" - hiding the rank that is
    the whole guarantee. Pin the corrected behaviour.
    """
    torch.manual_seed(0)
    U = orthonormalize(torch.randn(D, R))
    mu = torch.randn(D) * 20.0

    class AffineEraser(torch.nn.Module):
        def forward(self, z):
            zc = z - mu.to(z.dtype)
            return zc - (zc @ U.to(z.dtype)) @ U.to(z.dtype).T + mu.to(z.dtype)

    report = audit_eraser(AffineEraser(), D)
    assert report['linear'], "an affine map must be recognised, not called nonlinear"
    assert report['invertible'] is False
    assert report['rank'] == D - R
    assert report['erased_dims'] == R
    assert report['affine_offset'] > 1.0, "the mean offset should be reported"


def test_audit_still_flags_an_invertible_affine_map():
    """A nonzero bias must not let an invertible map pass as rank-deficient."""
    torch.manual_seed(0)
    b = torch.randn(D) * 5.0

    class ShiftOnly(torch.nn.Module):
        def forward(self, z):
            return z + b.to(z.dtype)

    report = audit_eraser(ShiftOnly(), D)
    assert report['linear']
    assert report['invertible'] is True
    assert report['rank'] == D


def test_spectral_lambda_zero_recovers_plain_svd():
    """lam=0 must reproduce the plain affine-SVD subspace exactly."""
    from src.unlearning.spectral import spectral_subspace
    torch.manual_seed(0)
    X = torch.randn(300, D) @ (torch.randn(D, D) * 0.3) + torch.randn(D) * 4.0

    U_svd = svd_subspace(X, k=R)
    U_spec = spectral_subspace(X, k=R, lam=0.0)

    # Same span (signs/order of eigenvectors are not canonical).
    P_svd = U_svd @ U_svd.T
    P_spec = U_spec @ U_spec.T
    assert torch.allclose(P_svd, P_spec, atol=1e-3), \
        f"lam=0 diverged from plain SVD (max diff {(P_svd - P_spec).abs().max():.2e})"


def test_spectral_avoids_directions_the_controls_occupy():
    """
    A direction with high variance in BOTH target and control should lose to a
    target-private one once lambda > 0. This is the whitening intent, without the
    matrix inverse that made whitening fail at n << d.
    """
    from src.unlearning.spectral import spectral_subspace
    torch.manual_seed(0)
    n, shared, private = 500, 0, 1

    Xt = torch.randn(n, D) * 0.1
    Xt[:, shared] += torch.randn(n) * 8.0
    Xt[:, private] += torch.randn(n) * 6.0

    Xc = torch.randn(n, D) * 0.1
    Xc[:, shared] += torch.randn(n) * 8.0

    U0 = spectral_subspace(Xt, k=1, controls=[Xc], lam=0.0)
    U1 = spectral_subspace(Xt, k=1, controls=[Xc], lam=1.0)

    assert U0[:, 0].abs().argmax().item() == shared, "lam=0 should take the loudest"
    assert U1[:, 0].abs().argmax().item() == private, "lam=1 should avoid the shared one"


def test_spectral_needs_no_matrix_inverse_at_small_n():
    """
    The regime that broke background whitening: fewer samples than dimensions, so
    the control covariance is singular. The pencil must stay finite and sane.
    """
    from src.unlearning.spectral import covariance_pencil, spectral_subspace
    torch.manual_seed(0)
    Xt = torch.randn(20, D)                    # n=20 << d=64
    Xc = torch.randn(20, D)

    M = covariance_pencil(Xt, [Xc], lam=1.0)
    assert torch.isfinite(M).all()
    U = spectral_subspace(Xt, k=4, controls=[Xc], lam=1.0)
    assert torch.isfinite(U).all()
    assert torch.allclose(U.T @ U, torch.eye(4), atol=1e-4)


def test_spectral_loss_is_differentiable_and_points_the_right_way():
    """
    Descending the loss must move the target cohort's variance OUT of span(U).

    Asserted as a relative decrease with a real margin, not against an arbitrary
    absolute level: how far it gets in a fixed budget is a property of the
    optimiser, whereas the DIRECTION is the property of the loss.
    """
    from src.unlearning.spectral import spectral_erasure_loss
    torch.manual_seed(0)
    U = orthonormalize(torch.randn(D, R))

    Zt = (torch.randn(200, D) * 0.3 + torch.randn(200, R) @ U.T).requires_grad_(True)
    Zc = torch.randn(200, D) * 0.3 + torch.randn(200, R) @ U.T

    def frac(Z):
        Zc_ = Z - Z.mean(0)
        return float((Zc_ @ U).pow(2).sum() / Zc_.pow(2).sum())

    before_loss = float(spectral_erasure_loss(Zt, U, [Zc], lam=1.0).detach())
    before_frac = frac(Zt.detach())

    opt = torch.optim.Adam([Zt], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        spectral_erasure_loss(Zt, U, [Zc], lam=1.0).backward()
        opt.step()

    after_loss = float(spectral_erasure_loss(Zt, U, [Zc], lam=1.0).detach())
    after_frac = frac(Zt.detach())

    assert after_loss < before_loss, f"loss did not decrease ({before_loss:.4f} -> {after_loss:.4f})"
    assert after_frac < before_frac * 0.5, (
        f"target variance along U barely moved ({before_frac:.4f} -> {after_frac:.4f})"
    )


def test_spectral_loss_leaves_control_variance_alone():
    """The control term must not be what drives the target's variance down."""
    from src.unlearning.spectral import spectral_erasure_loss
    torch.manual_seed(0)
    U = orthonormalize(torch.randn(D, R))
    Zt = (torch.randn(200, D) * 0.3 + torch.randn(200, R) @ U.T).requires_grad_(True)

    # No controls at all: the loss must still push the target's variance out.
    def frac(Z):
        Zc_ = Z - Z.mean(0)
        return float((Zc_ @ U).pow(2).sum() / Zc_.pow(2).sum())

    before = frac(Zt.detach())
    opt = torch.optim.Adam([Zt], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        spectral_erasure_loss(Zt, U).backward()
        opt.step()
    assert frac(Zt.detach()) < before * 0.5


def test_lora_starts_as_exact_identity():
    """
    B is initialised to zero, so a freshly wrapped model must be bit-identical to
    the original. Without this, "did poisoning change the encoder" is confounded by
    the wrapping itself.
    """
    from src.unlearning.lora import LoRALinear
    torch.manual_seed(0)
    base = torch.nn.Linear(D, 32)
    wrapped = LoRALinear(base, r=4)
    x = torch.randn(16, D)
    assert torch.allclose(base(x), wrapped(x), atol=1e-6)


def test_lora_trains_only_adapters():
    from src.unlearning.lora import freeze_base, inject_lora, lora_parameters
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(D, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8))
    for i, layer in enumerate(net):
        if isinstance(layer, torch.nn.Linear):
            layer._name_hint = f"q_proj{i}"
    # Wrap by explicit name match on the container's children.
    wrapped = inject_lora(net, target_names=('0', '2'), r=4, verbose=False)
    n_train, n_total = freeze_base(net)
    assert wrapped, "nothing was wrapped"
    assert 0 < n_train < n_total
    assert all(('.A' in n or '.B' in n) for n, p in net.named_parameters()
               if p.requires_grad)
    assert len(lora_parameters(net)) == 2 * len(wrapped)


def test_lora_gradients_flow_to_adapters_only():
    from src.unlearning.lora import LoRALinear
    torch.manual_seed(0)
    base = torch.nn.Linear(D, 16)
    w = LoRALinear(base, r=4)
    with torch.no_grad():
        w.B.normal_(0, 0.1)
    out = w(torch.randn(8, D)).pow(2).sum()
    out.backward()
    assert w.A.grad is not None and w.B.grad is not None
    assert w.base.weight.grad is None, "base weights must stay frozen"


class _Projector(torch.nn.Module):
    """Minimal non-invertible eraser, standing in for the deleted ProjectionEraser."""

    def __init__(self, U):
        super().__init__()
        self.register_buffer('U', U)

    def forward(self, z):
        U = self.U.to(z.dtype)
        return z - (z @ U) @ U.T

    def bottleneck_rank(self):
        return self.U.shape[0] - self.U.shape[1]


def test_audit_flags_a_projection_as_rank_deficient():
    torch.manual_seed(0)
    report = audit_eraser(_Projector(orthonormalize(torch.randn(D, R))), D)
    assert report['linear']
    assert report['invertible'] is False
    assert report['rank'] == D - R
    assert report['erased_dims'] == R


def test_projection_annihilates_its_subspace():
    torch.manual_seed(0)
    U = orthonormalize(torch.randn(D, R))
    er = _Projector(U)
    z = torch.randn(32, D)
    zp = er(z)
    assert (zp @ U).abs().max() < 1e-4
    assert torch.allclose(er(zp), zp, atol=1e-4)
    assert torch.allclose(remove_subspace(z, U), zp, atol=1e-5)


def test_projection_destroys_a_planted_signal():
    torch.manual_seed(0)
    basis = torch.zeros(D, R)
    basis[:R] = torch.eye(R)
    er = _Projector(basis)

    y = torch.randint(0, 2, (256,))
    z = torch.randn(256, D) * 0.1
    z[:, 0] += y.float() * 10.0

    assert linear_r2(z[:, :1], y.float()) > 0.9
    zp = er(z)
    assert zp[:, 0].abs().max() < 1e-4
    assert linear_r2(zp[:, :1], y.float()) < 0.05


def test_bottleneck_output_is_a_function_of_the_code_only():
    """
    The DPI guarantee rests on z' depending on z ONLY through the m-dim code.
    Two inputs with the same code must give the same output.
    """
    from src.unlearning.bottleneck import BottleneckEraser
    torch.manual_seed(0)
    er = BottleneckEraser(input_dim=D, code_dim=8, hidden=32)
    er.eval()
    z = torch.randn(16, D)
    with torch.no_grad():
        code = er.encode(z)
        # Decode the code directly; must match the full forward pass.
        assert torch.allclose(er.dec(code), er(z), atol=1e-6)


def test_bottleneck_declares_its_cap_to_the_audit():
    from src.unlearning.bottleneck import BottleneckEraser
    torch.manual_seed(0)
    er = BottleneckEraser(input_dim=D, code_dim=8, hidden=32)
    report = audit_eraser(er, D)
    assert report['invertible'] is False
    assert report['rank'] == 8


def test_reconstruction_objective_defeats_erasure():
    """
    NEGATIVE RESULT, pinned deliberately.

    A reconstruction term asks the model to PRESERVE z; an erasure term asks it to
    DESTROY part of z. Reconstruction wins, even on a signal purpose-built for the
    bottleneck to excel at.

    The fixture verifies itself first: the class sets the RADIUS in a 2-D plane, so
    both classes share a mean and every principal direction (no projection can
    reach it), the radius feature separates the classes (the signal exists), and no
    linear readout finds it (it is genuinely nonlinear).

    Measured: 0.973 -> 0.998 here, and on real Virchow2 features the autoencoder
    reconstructed 2560-d embeddings through a 64-d code at 2% error while erasing
    nothing (TCGA-LUNG 0.9335 -> 0.8211, PANDA 0.8496 -> 0.8672 i.e. worse than
    baseline). See AGENTS.md.

    If someone finds an objective that breaks this tension, THIS TEST SHOULD FAIL -
    that is the point of pinning it.
    """
    from src.unlearning.bottleneck import BottleneckEraser, fit_bottleneck
    torch.manual_seed(0)

    n = 3000
    y = torch.randint(0, 2, (n,))
    theta = torch.rand(n) * 6.2832
    radius = 2.0 + y.float() * 5.0
    z = torch.randn(n, D)
    z[:, 0] += radius * torch.cos(theta)
    z[:, 1] += radius * torch.sin(theta)
    tr, te = slice(0, 2200), slice(2200, n)

    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y.numpy(), z[:, :2].norm(dim=1).numpy()) > 0.95
    assert linear_r2(z[te], y[te].float()) < 0.25, "signal was linearly readable"

    def nonlinear_auc(Xtr, ytr, Xte, yte):
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        c = make_pipeline(StandardScaler(),
                          MLPClassifier((256, 128), max_iter=1200, random_state=0))
        c.fit(Xtr, ytr)
        return roc_auc_score(yte, c.predict_proba(Xte)[:, 1])

    before = nonlinear_auc(z[tr].numpy(), y[tr].numpy(), z[te].numpy(), y[te].numpy())
    assert before > 0.85, f"fixture broken: signal not recoverable ({before:.3f})"

    er = BottleneckEraser(input_dim=D, code_dim=16, hidden=128)
    fit_bottleneck(er, z[tr], y[tr], lambda_t=50.0, steps=600,
                   batch_size=256, verbose=False)
    with torch.no_grad():
        a_tr, a_te = er(z[tr]), er(z[te])

    after = nonlinear_auc(a_tr.numpy(), y[tr].numpy(), a_te.numpy(), y[te].numpy())
    # The documented outcome: erasure does NOT happen.
    assert after > before - 0.15, (
        f"bottleneck erased more than expected ({before:.3f} -> {after:.3f}); "
        "the reconstruction/erasure tension may have been broken - update AGENTS.md"
    )
