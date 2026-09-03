"""
What does the erased subspace actually contain?

Three analyses, each answering a question the paper needs:

A. PRINCIPAL ANGLES between subspaces fitted on different cohorts. If the top-k
   directions were a "concept direction" for, say, breast pathology, then the
   subspace fitted on BACH and the one fitted on BRACS would largely coincide.
   The transfer experiment says they do not; this measures how far apart they are
   and whether same-organ pairs are any closer than cross-organ pairs.

B. WHAT THE REMOVED COORDINATES PREDICT. Split each slide vector into its
   projection onto the removed subspace (k dims) and the surviving complement.
   Then ask what each half predicts: the clinical LABEL, or which COHORT the
   slide came from. If the removed half carries cohort identity as strongly as it
   carries label, the subspace is cohort geometry that happens to correlate with
   the label inside one cohort - which is exactly why it does not transfer.

C. UMAP of clean vs erased features, coloured by label and by cohort, to see
   which structure collapses and which survives.

Everything runs on the cached slide-mean features; no patch reads.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# umap-learn is a --user install, and the pytorch-conda module sets
# site.ENABLE_USER_SITE = False, so the user site directory is not on sys.path.
# Add it explicitly rather than silently skipping the figures.
_USER_SITE = os.path.expanduser('~/.local/lib/python3.11/site-packages')
if os.path.isdir(_USER_SITE) and _USER_SITE not in sys.path:
    sys.path.append(_USER_SITE)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.unlearning.subspace import remove_subspace_affine, svd_subspace  # noqa: E402

ORGAN = {'BACH': 'BREAST', 'BRACS': 'BREAST', 'TCGA-BRCA': 'BREAST',
         'PANDA': 'PROSTATE', 'TCGA-LUNG': 'LUNG', 'UBC-OCEAN': 'OVARIAN'}


def random_subspace(d, k, seed):
    """Orthonormal k-frame drawn uniformly, for the overlap null distribution."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, k, generator=g))
    return Q[:, :k]


def load(ds, fm, cache_dir, tag='nall'):
    p = f"{cache_dir}/{ds}_{fm}_p256_{tag}.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return dict(X=torch.tensor(z['X']).float(), y=z['labels'], pat=z['patients'])


def principal_angles(U1, U2):
    s = torch.linalg.svdvals(U1.double().T @ U2.double()).clamp(-1, 1)
    return float(s.mean()), float(s.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fm', default='features_virchow2')
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--cache_dir', default='results/quick/cache')
    ap.add_argument('--out_dir', default='results/interpret')
    ap.add_argument('--max_per_cohort', type=int, default=1200)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.RandomState(a.seed)
    out = {'fm': a.fm, 'k': a.k}

    cohorts = {}
    for ds in ORGAN:
        d = load(ds, a.fm, a.cache_dir)
        if d is None:
            print(f"  (no cache for {ds})")
            continue
        X = d['X']
        if len(X) > a.max_per_cohort:
            idx = rng.choice(len(X), a.max_per_cohort, replace=False)
            d = dict(X=X[idx], y=d['y'][idx], pat=d['pat'][idx])
        cohorts[ds] = d
        print(f"  {ds:11s} {tuple(d['X'].shape)}  organ={ORGAN[ds]}")

    # ---- A. principal angles between cohort-fitted subspaces --------------- #
    print("\n== A. subspace overlap between cohorts (mean principal cosine) ==")
    # Fit on a patient-disjoint TRAIN half only. Fitting on all slides and then
    # scoring the same slides is in-sample and inflates everything downstream.
    from src.utils.splits import patient_folds
    tr_mask = {}
    for ds, d in cohorts.items():
        try:
            train_p, _ = patient_folds(a.metadata, ds, 0)
            tr_mask[ds] = torch.tensor(np.isin(d['pat'], train_p))
        except Exception:
            tr_mask[ds] = torch.ones(len(d['X']), dtype=torch.bool)
    U = {ds: svd_subspace(d['X'][tr_mask[ds]], k=a.k) for ds, d in cohorts.items()}
    names = list(U)
    out['principal_angles'] = {}
    same_organ, cross_organ = [], []
    print(f"{'':12s}" + "".join(f"{n[:9]:>10s}" for n in names))
    for i in names:
        row = []
        for j in names:
            m, _ = principal_angles(U[i], U[j])
            row.append(m)
            if i != j:
                out['principal_angles'][f"{i}|{j}"] = m
                (same_organ if ORGAN[i] == ORGAN[j] else cross_organ).append(m)
        print(f"{i[:11]:12s}" + "".join(f"{v:10.3f}" for v in row))
    out['overlap_same_organ'] = same_organ
    out['overlap_cross_organ'] = cross_organ
    out['overlap_same_organ_mean'] = float(np.mean(same_organ)) if same_organ else None
    out['overlap_cross_organ_mean'] = float(np.mean(cross_organ)) if cross_organ else None

    # Null: two INDEPENDENT random k-frames in the same ambient dimension. Without
    # this, neither 0.47 nor 0.55 can be called high or low.
    dmb = next(iter(U.values())).shape[0]
    null = [principal_angles(random_subspace(dmb, a.k, 900 + i),
                             random_subspace(dmb, a.k, 5000 + i))[0] for i in range(30)]
    out['overlap_null_mean'] = float(np.mean(null))
    out['overlap_null_sd'] = float(np.std(null))

    print(f"\n  same-organ  pairs : {out['overlap_same_organ_mean']:.3f}  "
          f"(n={len(same_organ)//2} distinct pairs)")
    print(f"  cross-organ pairs : {out['overlap_cross_organ_mean']:.3f}  "
          f"(n={len(cross_organ)//2} distinct pairs)")
    print(f"  RANDOM null       : {out['overlap_null_mean']:.3f} +/- {out['overlap_null_sd']:.3f}"
          f"   (independent k-frames in R^{dmb})")
    if same_organ and cross_organ:
        try:
            from scipy.stats import mannwhitneyu
            u, pv = mannwhitneyu(same_organ[::2], cross_organ[::2], alternative='greater')
            out['overlap_same_vs_cross_p'] = float(pv)
            print(f"  same > cross?      p = {pv:.4f}  "
                  f"({'significant' if pv < .05 else 'NOT significant'})")
        except Exception as e:
            print(f"  (test unavailable: {e})")
    print("  Read: if same-organ sits near cross-organ and both sit far above the")
    print("  random null, cohorts share generic structure but nothing organ-specific.")

    # ---- B. what do the removed coordinates predict? ---------------------- #
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.model_selection import train_test_split
    print("\n== B. label vs cohort information, removed half vs surviving half ==")
    out['halves'] = {}
    for ds, d in cohorts.items():
        X, y = d['X'], d['y']
        classes = np.unique(y)
        if len(classes) < 2:
            continue
        Uk = U[ds]                      # already fitted on the train half only
        mu = X[tr_mask[ds]].mean(0)
        Xc = X - mu
        removed = (Xc @ Uk)                      # k coordinates that get deleted
        kept = remove_subspace_affine(X, Uk, mu)  # everything that survives
        yb = (y == classes[-1]).astype(int) if len(classes) == 2 else None
        # Score on the held-out half the subspace was NOT fitted on.
        tr = np.where(tr_mask[ds].numpy())[0]
        te = np.where(~tr_mask[ds].numpy())[0]
        if len(te) < 30 or len(np.unique(y[te])) < 2:
            tr, te = train_test_split(np.arange(len(X)), test_size=0.3,
                                      random_state=a.seed, stratify=y)

        def score(A, target, binary):
            m = LogisticRegression(max_iter=3000).fit(A[tr], target[tr])
            if binary:
                return roc_auc_score(target[te], m.predict_proba(A[te])[:, 1])
            return accuracy_score(target[te], m.predict(A[te]))

        tgt = yb if yb is not None else y
        binary = yb is not None
        r = {
            'label_from_removed': float(score(removed.numpy(), tgt, binary)),
            'label_from_kept': float(score(kept.numpy(), tgt, binary)),
        }
        out['halves'][ds] = r
        unit = 'AUC' if binary else 'acc'
        print(f"  {ds:11s} label {unit}: removed-half {r['label_from_removed']:.4f}"
              f"   surviving-half {r['label_from_kept']:.4f}")

    # cohort identity, pooled across cohorts, using ONE cohort's subspace
    print("\n  -- can the removed coordinates tell you which COHORT a slide is from? --")
    ref = 'BRACS' if 'BRACS' in cohorts else names[0]
    Xa = torch.cat([cohorts[n]['X'] for n in names])
    who = np.concatenate([[n] * len(cohorts[n]['X']) for n in names])
    mu_r = cohorts[ref]['X'].mean(0)
    rem_all = ((Xa - mu_r) @ U[ref]).numpy()
    kept_all = remove_subspace_affine(Xa, U[ref], mu_r).numpy()
    tr, te = train_test_split(np.arange(len(Xa)), test_size=0.3,
                              random_state=a.seed, stratify=who)
    for nm, A in (('removed-half', rem_all), ('surviving-half', kept_all)):
        m = LogisticRegression(max_iter=2000).fit(A[tr], who[tr])
        acc = accuracy_score(who[te], m.predict(A[te]))
        out[f'cohort_id_{nm}'] = float(acc)
        print(f"    {nm:15s} cohort accuracy {acc:.4f}   ({len(names)} cohorts, "
              f"chance {1/len(names):.3f})")

    # ---- C. figures ------------------------------------------------------ #
    print("\n== C. figures ==")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import umap
    except Exception as e:
        print(f"  skipped ({e})")
        json.dump(out, open(f"{a.out_dir}/interpret.json", 'w'), indent=2)
        return

    CAV = ("UMAP is a nonlinear embedding of a projection that removes variance, so "
           "apparent\nseparation is not evidence on its own - read every panel against "
           "the quoted AUC.")

    # -- Figure 1: subspace overlap between cohorts ------------------------- #
    M = np.array([[principal_angles(U[i], U[j])[0] for j in names] for i in names])
    f1, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(M, vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels([f"{n}\n({ORGAN[n]})" for n in names], fontsize=8, rotation=45,
                       ha='right')
    ax.set_yticklabels([f"{n} ({ORGAN[n]})" for n in names], fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha='center', va='center', fontsize=8,
                    color='white' if M[i, j] < .6 else 'black')
    ax.set_title("The erased subspace is cohort-specific, not organ- or concept-specific\n"
                 f"mean principal cosine between top-{a.k} subspaces fitted on each cohort",
                 fontsize=11)
    f1.colorbar(im, ax=ax, label="subspace overlap  (1.0 = identical)")
    f1.text(.01, .015, f"same-organ pairs {out['overlap_same_organ_mean']:.3f}   |   "
            f"cross-organ pairs {out['overlap_cross_organ_mean']:.3f}   |   "
            f"{a.fm.replace('features_','')}, slide-mean features", fontsize=8)
    f1.tight_layout(rect=[0, .04, 1, 1])
    p1 = f"{a.out_dir}/fig1_subspace_overlap_k{a.k}.png"
    f1.savefig(p1, dpi=150); print(f"  wrote {p1}")

    # -- Figure 2: what collapses inside one cohort ------------------------- #
    ds = 'TCGA-LUNG' if 'TCGA-LUNG' in cohorts else names[0]
    X, y = cohorts[ds]['X'], cohorts[ds]['y']
    Xe = remove_subspace_affine(X, U[ds], X.mean(0))
    hb = out['halves'].get(ds, {})
    f2, ax2 = plt.subplots(1, 2, figsize=(11.5, 5.4))
    for axi, (nm, A) in zip(ax2, (('before erasure', X.numpy()),
                                  ('after erasure', Xe.numpy()))):
        emb = umap.UMAP(n_neighbors=25, min_dist=.1, random_state=a.seed).fit_transform(A)
        for c in np.unique(y):
            m = y == c
            axi.scatter(emb[m, 0], emb[m, 1], s=5, alpha=.6, label=str(c))
        axi.set_title(nm, fontsize=11)
        axi.set_xlabel("UMAP-1"); axi.set_ylabel("UMAP-2")
        axi.set_xticks([]); axi.set_yticks([])
    ax2[0].legend(title=f"{ds} class", markerscale=3, fontsize=8, title_fontsize=8)
    f2.suptitle(f"Erasing the top-{a.k} subspace collapses the class structure it was "
                f"fitted on  ({ds})", fontsize=12)
    f2.text(.01, .015, CAV, fontsize=7.5)
    f2.tight_layout(rect=[0, .07, 1, .97])
    p2 = f"{a.out_dir}/fig2_class_collapse_{ds}_k{a.k}.png"
    f2.savefig(p2, dpi=150); print(f"  wrote {p2}")

    # -- Figure 3: cohort structure under a foreign subspace ---------------- #
    emb_raw = umap.UMAP(n_neighbors=25, min_dist=.1,
                        random_state=a.seed).fit_transform(Xa.numpy())
    emb_er = umap.UMAP(n_neighbors=25, min_dist=.1,
                       random_state=a.seed).fit_transform(kept_all)
    f3, ax3 = plt.subplots(1, 2, figsize=(12, 5.4))
    panels = ((ax3[0], emb_raw, "before erasure",
               out.get('cohort_id_removed-half')),
              (ax3[1], emb_er, f"after erasing the {ref}-fitted subspace",
               out.get('cohort_id_surviving-half')))
    for axi, emb, nm, _ in panels:
        for c in np.unique(who):
            m = who == c
            axi.scatter(emb[m, 0], emb[m, 1], s=5, alpha=.6, label=f"{c} ({ORGAN[c]})")
        axi.set_title(nm, fontsize=11)
        axi.set_xlabel("UMAP-1"); axi.set_ylabel("UMAP-2")
        axi.set_xticks([]); axi.set_yticks([])
    ax3[0].legend(title="cohort", markerscale=3, fontsize=7.5, title_fontsize=8)
    f3.suptitle("Cohort identity survives an erasure fitted on a different cohort",
                fontsize=12)
    f3.text(.01, .015,
            f"cohort recoverable from the {a.k} DELETED coordinates: "
            f"{out.get('cohort_id_removed-half', float('nan')):.3f} accuracy;  from "
            f"everything that survives: {out.get('cohort_id_surviving-half', float('nan')):.3f}"
            f"  (chance {1/len(names):.3f})", fontsize=8)
    f3.tight_layout(rect=[0, .06, 1, .97])
    p3 = f"{a.out_dir}/fig3_cohort_structure_k{a.k}.png"
    f3.savefig(p3, dpi=150); print(f"  wrote {p3}")

    json.dump(out, open(f"{a.out_dir}/interpret.json", 'w'), indent=2)
    print(f"  wrote {a.out_dir}/interpret.json")


if __name__ == '__main__':
    main()
