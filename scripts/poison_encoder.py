"""
Poison an encoder from the inside (plan section 19).

Instead of bolting a projection onto frozen embeddings, fine-tune LoRA adapters in
the encoder's last blocks so E_theta itself emits representations whose dominant
target-cohort subspace is uninformative:

    min_phi  tr( U^T Sigma_t(phi) U )  -  lambda * tr( U^T Sigma_c(phi) U )
             + mu * || Z(phi) - Z_0 ||^2 / || Z_0 ||^2

Two objectives are available (`--objective`):

  spectral  min_phi tr(U^T Sigma_t(phi) U), with U refit in closed form every
            `--refit_every` steps. MEASURED TO FAIL: it moved 61% -> 24% of the
            target variance out of span(U) exactly as designed, while the probe
            barely moved (0.9503 -> 0.9339). A projection destroys information
            because it is rank deficient; an ENCODER is a full-rank map, so told to
            avoid a subspace it simply relocates the signal. Optimising geometry
            does not erase.

  hsic      min_phi HSIC(Z(phi), y_target). Targets statistical DEPENDENCE rather
            than geometry, so relocating the signal does not help. Still not a
            min-max game - HSIC is a direct objective with no inner adversary,
            which matters because every adversarial variant here diverged.

Both add mu * ||Z - Z_0||^2 / ||Z_0||^2 to keep the encoder near its original
behaviour.

The released model then produces erased embeddings with no external transform,
which is the supply-chain threat model: nothing about the artefact looks unusual.

DATA REQUIREMENT
----------------
Needs raw patches. The pre-extracted HDF5 features cannot be used - the whole point
is to backprop into the encoder. Supply an image folder via --image_root, laid out
as <root>/<label>/<file>. On this cluster the original BACH/BRACS images are NOT
present (only extracted features and contour thumbnails), so run this where the
raw patches live. --smoke_test validates the machinery on synthetic images.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.unlearning.losses import hsic
from src.unlearning.lora import freeze_base, inject_lora, lora_parameters
from src.unlearning.spectral import spectral_erasure_loss, spectral_subspace


# --------------------------------------------------------------------------- #

def load_encoder(model_path, device):
    """Load a CLIP-style vision encoder and return (module, forward_fn, blocks)."""
    from transformers import CLIPModel
    model = CLIPModel.from_pretrained(model_path)
    vision = model.vision_model.to(device)

    def embed(pixel_values):
        out = vision(pixel_values=pixel_values)
        return out.pooler_output

    return vision, embed, vision.encoder.layers


def embed_all(embed, images, batch_size, device, grad=False):
    outs = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for i in range(0, len(images), batch_size):
            outs.append(embed(images[i:i + batch_size].to(device)))
    return torch.cat(outs, dim=0)


def make_synthetic(n, n_classes, image_size, seed=0, difficulty=0.12):
    """
    Synthetic patches with a DELIBERATELY WEAK class signal.

    An earlier version produced baseline AUC 1.0000, which makes an erasure result
    meaningless - anything looks like it works when the starting point is perfect
    separation. `difficulty` scales the class-carrying pattern against per-image
    noise; the aim is a baseline in the 0.80-0.95 range, comparable to the real
    tasks.

    Used only by --smoke_test, to show the loss backpropagates through a real ViT
    and moves the embedding spectrum. It says nothing about pathology.
    """
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, n_classes, (n,), generator=g)

    coords = torch.linspace(-1, 1, image_size)
    xx = coords.view(1, -1).expand(image_size, image_size)
    yy = xx.T

    imgs = torch.rand(n, 3, image_size, image_size, generator=g) * 0.5 + 0.25
    for c in range(n_classes):
        m = y == c
        if not m.any():
            continue
        phase = 3.0 + c * 2.3
        pattern = torch.sin(phase * xx + 0.7 * c) * torch.cos((c + 1.5) * yy)
        # Per-image random amplitude and channel weighting so the class is a
        # statistical tendency rather than a deterministic template.
        amp = difficulty * (0.5 + torch.rand(int(m.sum()), 1, 1, 1, generator=g))
        chan = torch.rand(int(m.sum()), 3, 1, 1, generator=g) * 0.5 + 0.75
        imgs[m] += amp * chan * pattern.unsqueeze(0).unsqueeze(0)

    imgs += torch.randn(n, 3, image_size, image_size, generator=g) * 0.08
    return imgs.clamp(0, 1), y


def load_image_folder(root, image_size, max_per_class=None):
    from PIL import Image
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    imgs, labels = [], []
    for ci, c in enumerate(classes):
        files = sorted(os.listdir(os.path.join(root, c)))
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            try:
                im = Image.open(os.path.join(root, c, f)).convert('RGB').resize(
                    (image_size, image_size))
            except Exception:
                continue
            imgs.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0)
            labels.append(ci)
    if not imgs:
        raise SystemExit(f"No readable images under {root}")
    return torch.stack(imgs), torch.tensor(labels), classes


def probe_auc(Z_tr, y_tr, Z_te, y_te, n_classes, seed=0):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.evaluation.metrics import macro_ovr_auc

    clf = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(256,), max_iter=400,
                                      random_state=seed, early_stopping=True))
    clf.fit(Z_tr, y_tr)
    p = clf.predict_proba(Z_te)
    full = np.zeros((len(Z_te), n_classes))
    for j, c in enumerate(clf[-1].classes_):
        full[:, int(c)] = p[:, j]
    auc, _ = macro_ovr_auc(y_te, full[:, 1] if n_classes == 2 else full,
                           n_classes, strict=False)
    return auc


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model_path', default='/work/hdd/bhwm/dchanda/model_cache/plip')
    ap.add_argument('--image_root', default=None,
                    help="<root>/<label>/<file.png>; omit with --smoke_test")
    ap.add_argument('--smoke_test', action='store_true',
                    help="validate the machinery on synthetic images")
    ap.add_argument('--max_per_class', type=int, default=None)
    ap.add_argument('--n_synthetic', type=int, default=256)
    ap.add_argument('--n_classes', type=int, default=4)
    ap.add_argument('--difficulty', type=float, default=0.12,
                    help="smoke test: class-signal strength; lower = harder. "
                         "Aim for a baseline AUC of 0.80-0.95, not 1.0")

    ap.add_argument('--objective', choices=['hsic', 'spectral'], default='hsic',
                    help="'hsic' targets dependence (recommended); 'spectral' "
                         "targets geometry and was measured to fail")
    ap.add_argument('--k', type=int, default=32, help="subspace rank for diagnostics")
    ap.add_argument('--lora_r', type=int, default=8)
    ap.add_argument('--lora_alpha', type=float, default=16.0)
    ap.add_argument('--last_n_blocks', type=int, default=4)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--refit_every', type=int, default=25,
                    help="re-solve U from current embeddings every N steps")
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lambda_c', type=float, default=0.0)
    ap.add_argument('--mu_fidelity', type=float, default=1.0,
                    help="weight on staying close to the original embeddings")
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/quick/poison_encoder.json')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}  model={args.model_path}")

    vision, embed, blocks = load_encoder(args.model_path, device)
    image_size = vision.config.image_size

    if args.smoke_test or args.image_root is None:
        print(f"SMOKE TEST: synthetic images ({args.n_synthetic}, "
              f"{args.n_classes} classes)")
        images, y = make_synthetic(args.n_synthetic, args.n_classes, image_size,
                                   args.seed, difficulty=args.difficulty)
        n_classes = args.n_classes
    else:
        images, y, classes = load_image_folder(args.image_root, image_size,
                                               args.max_per_class)
        n_classes = len(classes)
        print(f"loaded {len(images)} images, {n_classes} classes: {classes}")

    n = len(images)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_tr = int(0.7 * n)
    tr, te = perm[:n_tr], perm[n_tr:]

    # ---- baseline ------------------------------------------------------- #
    vision.eval()
    Z0 = embed_all(embed, images, args.batch_size, device).cpu()
    auc_before = probe_auc(Z0[tr].numpy(), y[tr].numpy(),
                           Z0[te].numpy(), y[te].numpy(), n_classes)
    print(f"\nbaseline embedding dim {Z0.shape[1]} | probe AUC {auc_before:.4f}")

    # ---- inject adapters -------------------------------------------------- #
    inject_lora(vision, target_names=('q_proj', 'v_proj'),
                last_n_blocks=args.last_n_blocks, block_container=blocks,
                r=args.lora_r, alpha=args.lora_alpha)
    vision.to(device)          # ensure adapters share the model's device
    n_train, n_total = freeze_base(vision)
    print(f"  trainable {n_train:,} / {n_total:,} params "
          f"({100 * n_train / n_total:.3f}%)")

    params = lora_parameters(vision)
    opt = torch.optim.Adam(params, lr=args.lr)
    Z0_dev = Z0.to(device)
    images_tr = images[tr]
    y_tr = y[tr]
    U = None

    # U characterises the target's dominant subspace; under --objective hsic it is
    # only used for the variance diagnostic, not for the loss.
    with torch.no_grad():
        U = spectral_subspace(Z0, k=args.k).to(device)

    vision.train()
    for step in range(args.steps):
        if args.objective == 'spectral' and step % args.refit_every == 0:
            with torch.no_grad():
                Z_now = embed_all(embed, images, args.batch_size, device)
            U = spectral_subspace(Z_now.cpu(), k=args.k).to(device)

        idx = torch.randperm(len(images_tr))[:args.batch_size]
        batch = images_tr[idx].to(device)
        opt.zero_grad()

        Z = embed(batch)
        if args.objective == 'hsic':
            loss_erase = hsic(Z, y_tr[idx].to(device))
        else:
            loss_erase = spectral_erasure_loss(Z, U, lam=0.0)
        target = Z0_dev[tr][idx]
        loss_fid = (Z - target).pow(2).sum() / target.pow(2).sum().clamp_min(1e-12)
        loss = loss_erase + args.mu_fidelity * loss_fid
        loss.backward()
        opt.step()

        if step % max(args.steps // 6, 1) == 0 or step == args.steps - 1:
            print(f"  step {step:4d} | {args.objective} {float(loss_erase):.5f} "
                  f"| fidelity {float(loss_fid):.5f}", flush=True)

    # ---- evaluate --------------------------------------------------------- #
    vision.eval()
    Z1 = embed_all(embed, images, args.batch_size, device).cpu()
    auc_after = probe_auc(Z1[tr].numpy(), y[tr].numpy(),
                          Z1[te].numpy(), y[te].numpy(), n_classes)

    cos = float(torch.nn.functional.cosine_similarity(Z0, Z1, dim=-1).mean())
    # Raw cosine is dominated by the embeddings' large shared mean, so it can look
    # near-perfect while the informative (centred) part has moved a lot. Same
    # anisotropy that made the raw SVD projection appear catastrophic at cos 0.43
    # until the mean-preserving fix. Report both.
    Z0c = Z0 - Z0.mean(0, keepdim=True)
    Z1c = Z1 - Z1.mean(0, keepdim=True)
    cos_centered = float(torch.nn.functional.cosine_similarity(Z0c, Z1c, dim=-1).mean())
    mean_frac = float(Z0.mean(0).pow(2).sum() * len(Z0) / Z0.pow(2).sum())
    with torch.no_grad():
        Zc0 = Z0 - Z0.mean(0)
        Zc1 = Z1 - Z1.mean(0)
        Uc = U.cpu().to(Zc0.dtype)
        frac0 = float((Zc0 @ Uc).pow(2).sum() / Zc0.pow(2).sum())
        frac1 = float((Zc1 @ Uc).pow(2).sum() / Zc1.pow(2).sum())

    print(f"\n{'=' * 62}")
    print(f"probe AUC        {auc_before:.4f} -> {auc_after:.4f}   (chance 0.50)")
    print(f"variance in U    {frac0:.4f} -> {frac1:.4f}")
    print(f"cos(z0, z1)      {cos:.4f}   (raw - inflated by the shared mean)")
    print(f"cos centred      {cos_centered:.4f}   <- the honest fidelity number")
    print(f"mean energy frac {mean_frac:.4f}   (how much of ||Z||^2 is the shared mean)")
    print(f"{'=' * 62}")
    print("NOTE: no external transform is applied at evaluation - the encoder "
          "itself now emits these embeddings.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({'auc_before': auc_before, 'auc_after': auc_after,
               'variance_in_U_before': frac0, 'variance_in_U_after': frac1,
               'cosine': cos, 'cosine_centered': cos_centered,
               'mean_energy_fraction': mean_frac, 'trainable_params': n_train,
               'total_params': n_total, 'args': vars(args)},
              open(args.out, 'w'), indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
