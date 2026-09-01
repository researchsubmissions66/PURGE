"""
Sanity checks for the erasure result.

Each check is designed to FAIL if the headline result is an artifact:

  1 leakage      train/test patients disjoint, and the eraser saw only train data
  2 random       a RANDOM subspace of the same rank must NOT kill the target;
                 if it does, the result is "deleting dimensions hurts", not
                 targeted erasure
  3 wrong-target a subspace fitted on a DIFFERENT cohort must NOT kill the target
  4 geometry     the projection is genuinely rank-deficient and the target's
                 class separation along the erased directions is actually gone
  5 chance       the post-erasure AUC is statistically indistinguishable from
                 chance (permutation null), not merely "low"
  6 capacity     a much stronger probe still cannot recover the target
  7 distortion   the embeddings are only mildly perturbed (not destroyed)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.evaluation.metrics import macro_ovr_auc
from src.evaluation.tasks import get_task
from src.unlearning.subspace import orthonormalize
from src.unlearning.subspace import remove_subspace_affine, svd_subspace
from src.utils.splits import patient_folds

OK, BAD = "PASS", "**FAIL**"


def probe_auc(X_tr, y_tr, X_te, y_te, n_cls, kind='mlp', seed=0, strong=False):
    if kind == 'logreg':
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight='balanced'))
    else:
        hidden = (1024, 512) if strong else (256,)
        clf = make_pipeline(StandardScaler(),
                            MLPClassifier(hidden_layer_sizes=hidden,
                                          max_iter=1500 if strong else 400,
                                          random_state=seed,
                                          early_stopping=not strong,
                                          n_iter_no_change=15,
                                          alpha=1e-5 if strong else 1e-4))
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)
    seen = clf[-1].classes_
    full = np.zeros((len(X_te), n_cls))
    for j, c in enumerate(seen):
        full[:, int(c)] = p[:, j]
    auc, _ = macro_ovr_auc(y_te, full[:, 1] if n_cls == 2 else full, n_cls, strict=False)
    return auc


def load_task(task_name, cache_dir, fm, metadata, fold, max_slides=600):
    spec = get_task(task_name)
    ds = spec['dataset']
    cache = os.path.join(cache_dir, f"{ds}_{fm}_p256_n{max_slides}.npz")
    if not os.path.exists(cache):
        raise SystemExit(f"missing cache {cache}")
    d = np.load(cache, allow_pickle=True)
    X, patients = d['X'], d['patients']
    if 'labels' in d:
        labels = d['labels']
    else:
        # Caches written before the task refactor stored mapped indices.
        labels = np.array([list(d['classes'])[i] for i in d['y']], dtype=object)

    m = spec['map']
    keep = np.array([str(l) in m for l in labels])
    X, y, p = X[keep], np.array([m[str(l)] for l in labels[keep]]), patients[keep]

    train_p, test_p = patient_folds(metadata, ds, fold)
    tr, te = np.isin(p, train_p), np.isin(p, test_p)
    return dict(Xtr=X[tr], ytr=y[tr], Xte=X[te], yte=y[te],
                ptr=p[tr], pte=p[te], n_cls=len(set(m.values())), dataset=ds)


def project_out(X, U):
    t = torch.tensor(X)
    return (t - (t @ U) @ U.T).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', default='TCGA-LUNG-subtype')
    ap.add_argument('--other_task', default='UBC-OCEAN-subtype',
                    help="cohort whose subspace should NOT erase the target")
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--fm', default='features_virchow2')
    ap.add_argument('--metadata', default='data/multi_benchmark_metadata.csv')
    ap.add_argument('--cache_dir', default='results/quick/cache')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--n_random', type=int, default=5)
    ap.add_argument('--n_perm', type=int, default=200)
    ap.add_argument('--out', default='results/quick/sanity.json')
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    T = load_task(args.task, args.cache_dir, args.fm, args.metadata, args.fold)
    O = load_task(args.other_task, args.cache_dir, args.fm, args.metadata, args.fold)
    d = T['Xtr'].shape[1]
    report, verdicts = {}, {}

    print(f"TASK {args.task}  d={d}  k={args.k}")
    print(f"  train {len(T['Xtr'])} / test {len(T['Xte'])}  ({T['n_cls']} classes)\n")

    # -- 1 leakage -------------------------------------------------------- #
    overlap = set(T['ptr']) & set(T['pte'])
    verdicts['1_no_patient_leakage'] = OK if not overlap else BAD
    print(f"[1] train/test patient overlap: {len(overlap)}  -> {verdicts['1_no_patient_leakage']}")

    # -- baseline & targeted erasure -------------------------------------- #
    base = probe_auc(T['Xtr'], T['ytr'], T['Xte'], T['yte'], T['n_cls'])
    U_t = svd_subspace(torch.tensor(T['Xtr']), k=args.k)
    tgt = probe_auc(project_out(T['Xtr'], U_t), T['ytr'],
                    project_out(T['Xte'], U_t), T['yte'], T['n_cls'])
    report.update(baseline=base, targeted=tgt)
    print(f"\n    baseline AUC            {base:.4f}")
    print(f"    after targeted erasure  {tgt:.4f}")

    # -- 2 random subspace of the same rank ------------------------------- #
    rand = []
    for i in range(args.n_random):
        U_r = orthonormalize(torch.randn(d, args.k, generator=torch.Generator().manual_seed(i)))
        rand.append(probe_auc(project_out(T['Xtr'], U_r), T['ytr'],
                              project_out(T['Xte'], U_r), T['yte'], T['n_cls'], seed=i))
    report['random_subspace'] = rand
    rm, rs = float(np.mean(rand)), float(np.std(rand))
    # A random subspace must leave the task essentially intact.
    verdicts['2_random_subspace_harmless'] = OK if rm > base - 0.05 else BAD
    print(f"\n[2] random rank-{args.k} subspace: {rm:.4f} +/- {rs:.4f} "
          f"(vs baseline {base:.4f})  -> {verdicts['2_random_subspace_harmless']}")
    print(f"    targeted {tgt:.4f} must be far below random {rm:.4f}: "
          f"gap {rm - tgt:+.4f}")

    # -- 3 subspace fitted on a different cohort -------------------------- #
    U_o = svd_subspace(torch.tensor(O['Xtr']), k=args.k)
    wrong = probe_auc(project_out(T['Xtr'], U_o), T['ytr'],
                      project_out(T['Xte'], U_o), T['yte'], T['n_cls'])
    report['wrong_cohort_subspace'] = wrong
    verdicts['3_wrong_cohort_harmless'] = OK if wrong > base - 0.10 else BAD
    print(f"\n[3] subspace from {args.other_task}: {wrong:.4f} "
          f"(vs baseline {base:.4f})  -> {verdicts['3_wrong_cohort_harmless']}")

    # -- 4 geometry -------------------------------------------------------- #
    P = (torch.eye(d, dtype=torch.float64)
         - U_t.double() @ U_t.double().T)
    s = torch.linalg.svdvals(P)
    # A float32 orthonormal basis satisfies |U^T U - I| ~ 1e-6, which puts the
    # "zero" singular values of P around 3e-6. A 1e-6 relative cutoff therefore
    # sits on the noise floor and miscounts; 1e-4 separates cleanly.
    rank = int((s > 1e-4 * s[0]).sum())
    Xe = project_out(T['Xte'], U_t)
    resid = float(np.abs(Xe @ U_t.numpy()).max())
    # class separation along the erased directions, before vs after
    def sep(X, y):
        cs = [X[y == c].mean(0) for c in np.unique(y)]
        return float(np.linalg.norm(np.stack(cs) - np.mean(cs, 0), axis=1).mean())
    sep_before = sep(T['Xte'] @ U_t.numpy(), T['yte'])
    sep_after = sep(Xe @ U_t.numpy(), T['yte'])
    report.update(rank=rank, residual=resid, sep_before=sep_before, sep_after=sep_after)
    resid_before = float(np.abs(T['Xte'] @ U_t.numpy()).max())
    ratio = resid / max(resid_before, 1e-12)
    report['residual_ratio'] = ratio
    verdicts['4_rank_deficient'] = OK if (rank == d - args.k and ratio < 1e-4) else BAD
    print(f"\n[4] rank(P) = {rank} (expected {d - args.k})  -> {verdicts['4_rank_deficient']}")
    print(f"    residual in erased subspace: {resid_before:.3e} -> {resid:.3e} "
          f"(ratio {ratio:.2e})")
    print(f"    class separation along erased dirs: {sep_before:.4f} -> {sep_after:.3e}")

    # -- 5 permutation null ------------------------------------------------ #
    rng = np.random.RandomState(0)
    null = []
    Xe_tr = project_out(T['Xtr'], U_t)
    for _ in range(args.n_perm):
        yp = rng.permutation(T['yte'])
        null.append(macro_ovr_auc(
            yp, rng.rand(len(yp)) if T['n_cls'] == 2 else rng.dirichlet(
                np.ones(T['n_cls']), len(yp)), T['n_cls'], strict=False)[0])
    null = np.array([x for x in null if x is not None])
    lo, hi = np.percentile(null, [2.5, 97.5])
    report['null_ci'] = [float(lo), float(hi)]
    verdicts['5_indistinguishable_from_chance'] = OK if lo <= tgt <= hi else f"outside null [{lo:.3f},{hi:.3f}]"
    print(f"\n[5] chance null 95% CI: [{lo:.4f}, {hi:.4f}]; erased AUC {tgt:.4f}"
          f"  -> {verdicts['5_indistinguishable_from_chance']}")

    # -- 6 much stronger probe --------------------------------------------- #
    strong = probe_auc(Xe_tr, T['ytr'], Xe, T['yte'], T['n_cls'], strong=True)
    strong_base = probe_auc(T['Xtr'], T['ytr'], T['Xte'], T['yte'], T['n_cls'], strong=True)
    report.update(strong_erased=strong, strong_baseline=strong_base)
    verdicts['6_strong_probe_fails'] = OK if strong < base - 0.20 else BAD
    print(f"\n[6] high-capacity probe (1024-512): baseline {strong_base:.4f}, "
          f"erased {strong:.4f}  -> {verdicts['6_strong_probe_fails']}")

    # -- 7 distortion ------------------------------------------------------- #
    num = np.linalg.norm(T['Xte'] - Xe, axis=1)
    den = np.linalg.norm(T['Xte'], axis=1)
    frac = float(np.mean(num / den))
    cos = float(np.mean(np.sum(T['Xte'] * Xe, 1) /
                        (np.linalg.norm(T['Xte'], axis=1) * np.linalg.norm(Xe, axis=1))))
    report.update(rel_change=frac, cosine=cos)
    verdicts['7_embedding_mostly_intact'] = OK if cos > 0.90 else BAD
    print(f"\n[7] plain projection: ||z-z'||/||z|| = {frac:.4f}; "
          f"cos(z,z') = {cos:.4f}  -> {verdicts['7_embedding_mostly_intact']}")

    # -- 8 affine (mean-preserving) variant -------------------------------- #
    mu = torch.tensor(T['Xtr']).mean(0)
    Xa_tr = remove_subspace_affine(torch.tensor(T['Xtr']), U_t, mu).numpy()
    Xa_te = remove_subspace_affine(torch.tensor(T['Xte']), U_t, mu).numpy()
    aff = probe_auc(Xa_tr, T['ytr'], Xa_te, T['yte'], T['n_cls'])
    aff_strong = probe_auc(Xa_tr, T['ytr'], Xa_te, T['yte'], T['n_cls'], strong=True)
    na = np.linalg.norm(T['Xte'] - Xa_te, axis=1)
    frac_a = float(np.mean(na / den))
    cos_a = float(np.mean(np.sum(T['Xte'] * Xa_te, 1) /
                          (np.linalg.norm(T['Xte'], axis=1) * np.linalg.norm(Xa_te, axis=1))))
    report.update(affine_auc=aff, affine_strong=aff_strong,
                  affine_rel_change=frac_a, affine_cosine=cos_a)
    verdicts['8_affine_erases_as_well'] = OK if aff < base - 0.20 else BAD
    verdicts['9_affine_low_distortion'] = OK if cos_a > 0.90 else BAD
    print(f"\n[8] AFFINE variant: AUC {aff:.4f} (plain {tgt:.4f}, baseline {base:.4f})"
          f"  -> {verdicts['8_affine_erases_as_well']}")
    print(f"    strong probe on affine: {aff_strong:.4f}")
    print(f"\n[9] affine distortion: ||z-z'||/||z|| = {frac_a:.4f}; "
          f"cos(z,z') = {cos_a:.4f}  -> {verdicts['9_affine_low_distortion']}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    for k, v in verdicts.items():
        print(f"  {k:38s} {v}")
    failed = [k for k, v in verdicts.items() if v != OK]
    print("\n" + ("ALL CHECKS PASSED" if not failed else f"FAILURES: {failed}"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({'task': args.task, 'k': args.k, 'report': report, 'verdicts': verdicts},
              open(args.out, 'w'), indent=2)


if __name__ == "__main__":
    main()
