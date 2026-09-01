"""
Fast first-pass validation of the erasure idea on mean-pooled slide features.

Why pooled features are a fair proxy
------------------------------------
Every eraser here is linear, and mean pooling is linear, so they commute:

    mean_i(P z_i) = P(mean_i z_i)

A probe on erased slide-means is therefore EXACTLY MeanMIL on erased patches, and
a close approximation of ABMIL (whose attention weights would shift slightly).
This turns hours of MIL training into minutes, at the cost of not capturing
attention re-weighting - so treat a positive result here as "worth running the
full MIL sweep", not as the final number.

What it reports
---------------
For each method: the AUC of probes RETRAINED FROM SCRATCH on the erased features,
for the target task and every control task, plus the selective degradation score.
Chance for macro one-vs-rest AUC is 0.50 for any class count.
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.datasets.feature_dataset import build_label_map, select_dataset, to_h5_name
from src.evaluation.metrics import macro_ovr_auc, selective_degradation_score
from src.evaluation.tasks import TASKS, get_task, relation
from src.unlearning.subspace import remove_subspace_affine, svd_subspace
from src.unlearning.spectral import spectral_subspace
from src.utils.splits import patient_folds


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def slab_starts(n, patches, block):
    """
    Evenly spaced contiguous block starts covering the patch axis.

    Features are chunked (1, D) on disk, so 256 scattered row reads means 256
    random chunk reads - on a contended parallel filesystem that dominates
    everything. Reading a few contiguous slabs instead keeps the sample spread
    across the slide while turning it into a handful of sequential reads.
    """
    if n <= patches:
        return None
    n_blocks = max(1, patches // block)
    return np.unique(np.linspace(0, max(n - block, 0), n_blocks).astype(int))


def load_pooled(dataset, encoder_dir, metadata, patches, workers, cache_dir,
                seed=0, block=32, max_slides=None):
    """Mean-pool each slide over a strided patch subsample. Cached to .npz."""
    tag = f"{dataset}_{os.path.basename(encoder_dir)}_p{patches}"
    if max_slides:
        tag += f"_n{max_slides}"
    cache = os.path.join(cache_dir, tag + ".npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if 'labels' in d:
            labels = d['labels']
        else:
            # Back-compat with caches that stored mapped indices.
            labels = np.array([list(d['classes'])[i] for i in d['y']], dtype=object)
        print(f"  {dataset}: {d['X'].shape[0]} slides (cached)", flush=True)
        return d['X'], labels, d['patients']

    df = select_dataset(pd.read_csv(metadata), dataset).reset_index(drop=True)
    label_map = build_label_map(metadata, dataset)
    classes = sorted(label_map, key=label_map.get)

    if max_slides and len(df) > max_slides:
        # Stratify the cap by label so rare classes survive.
        df = (df.groupby('label', group_keys=False)
                .apply(lambda g: g.sample(
                    max(1, int(round(max_slides * len(g) / len(df)))),
                    random_state=seed))
                .reset_index(drop=True))

    def load(row):
        path = os.path.join(encoder_dir, to_h5_name(row['filename']))
        try:
            with h5py.File(path, 'r') as h:
                feats = h['features']
                n = feats.shape[0]
                starts = slab_starts(n, patches, block)
                if starts is None:
                    return feats[:].mean(0)
                total = np.zeros(feats.shape[1], dtype=np.float64)
                count = 0
                for st in starts:
                    blk = feats[st:min(st + block, n)]
                    total += blk.sum(0)
                    count += len(blk)
                return (total / max(count, 1)).astype(np.float32)
        except (OSError, KeyError):
            return None

    t0 = time.time()
    rows = [r for _, r in df.iterrows()]
    done = [0]

    def load_reporting(row):
        out = load(row)
        done[0] += 1
        if done[0] % 200 == 0:
            print(f"    {dataset}: {done[0]}/{len(rows)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        return out

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        vecs = list(pool.map(load_reporting, rows))

    X, labels, patients = [], [], []
    for vec, row in zip(vecs, rows):
        if vec is not None:
            X.append(vec)
            labels.append(str(row['label']))
            patients.append(row['patient_id'])

    if not X:
        raise SystemExit(f"No readable features for {dataset} under {encoder_dir}")

    X = np.stack(X).astype(np.float32)
    labels = np.array(labels, dtype=object)
    patients = np.asarray(patients)
    print(f"  {dataset}: {X.shape[0]}/{len(rows)} slides, d={X.shape[1]}, "
          f"{time.time() - t0:.0f}s", flush=True)

    os.makedirs(cache_dir, exist_ok=True)
    tmp = f"{cache}.tmp.{os.getpid()}.npz"
    np.savez(tmp, X=X, labels=labels, patients=patients,
             classes=np.array(classes, dtype=object))
    os.replace(tmp, cache)
    return X, labels, patients


def build_task(task_name, dataset_cache, metadata, fold):
    """
    Materialize one task from its dataset's cached features.

    The patient split is taken from the DATASET, not the task, so two tasks over
    the same slides share it exactly. Otherwise a slide could be in the target's
    training set and the control's test set, and the eraser would have seen the
    control's test data during fitting.
    """
    spec = get_task(task_name)
    X, labels, patients = dataset_cache[spec['dataset']]
    mapping = spec['map']

    keep = np.array([str(l) in mapping for l in labels])
    Xk = X[keep]
    yk = np.array([mapping[str(l)] for l in labels[keep]])
    pk = patients[keep]

    train_p, test_p = patient_folds(metadata, spec['dataset'], fold)
    tr = np.isin(pk, train_p)
    te = np.isin(pk, test_p)
    n_classes = len(set(mapping.values()))

    if tr.sum() == 0 or te.sum() == 0:
        raise SystemExit(f"Task {task_name}: empty split "
                         f"(train={tr.sum()}, test={te.sum()})")
    return Xk[tr], yk[tr], Xk[te], yk[te], n_classes


def fresh_probe_auc(X_tr, y_tr, X_te, y_te, n_classes, kind, seed=0):
    if kind == 'logreg':
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0),
        )
    elif kind == 'mlp':
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256,), max_iter=400, random_state=seed,
                          early_stopping=True, n_iter_no_change=15),
        )
    else:
        raise ValueError(kind)

    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)

    # Map predict_proba columns (over classes seen in training) onto all classes.
    seen = clf.classes_ if hasattr(clf, 'classes_') else clf[-1].classes_
    full = np.zeros((len(X_te), n_classes))
    for j, c in enumerate(seen):
        full[:, int(c)] = probs[:, j]

    if n_classes == 2:
        auc, note = macro_ovr_auc(y_te, full[:, 1], 2, strict=False)
    else:
        auc, note = macro_ovr_auc(y_te, full, n_classes, strict=False)
    return auc, note


def evaluate(transform, tasks, probes):
    """Retrain every probe family from scratch on the transformed features."""
    out = {}
    for name, (X_tr, y_tr, X_te, y_te, n_cls) in tasks.items():
        aucs = {}
        for kind in probes:
            auc, _ = fresh_probe_auc(
                transform(X_tr), y_tr, transform(X_te), y_te, n_cls, kind
            )
            aucs[kind] = auc
        out[name] = aucs
    return out


def show(res, order):
    for name in order:
        a = res.get(name, {})
        print("  " + f"{name:22s} " + "  ".join(
            f"{k}={v:.4f}" if v is not None else f"{k}=n/a" for k, v in a.items()), flush=True)


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target_task', action='append', default=[],
                        help="task to erase; repeatable (each becomes a matrix row)")
    parser.add_argument('--control_task', action='append', default=[],
                        help="task to preserve; repeatable")
    parser.add_argument('--fm', default='features_virchow2')
    parser.add_argument('--base_dir', default='/work/hdd/bhwm')
    parser.add_argument('--metadata', default='data/multi_benchmark_metadata.csv')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--patches', type=int, default=256)
    parser.add_argument('--block', type=int, default=32)
    parser.add_argument('--max_slides', type=int, default=None)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--cache_dir', default='results/quick/cache')
    parser.add_argument('--out', default='results/quick/quick_validate.json')

    parser.add_argument('--methods', nargs='+',
                        default=['svd', 'spectral', 'leace'],
                        help="svd (affine null-space, the base method), "
                             "spectral (control-aware generalisation), leace, "
                             "low_rank (invertible negative control)")
    parser.add_argument('--svd_ranks', type=int, nargs='+', default=[16, 64, 256])
    parser.add_argument('--lr_probe', type=float, default=1e-3)
    parser.add_argument('--probes', nargs='+', default=['logreg', 'mlp'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--spectral_lam', type=float, nargs='+',
                        default=[0.0, 0.5, 1.0, 2.0],
                        help="control-variance weight in the pencil; 0 = plain SVD")
    parser.add_argument('--no_svd_affine', action='store_true',
                        help="ablation: plain projection instead of the "
                             "mean-preserving default")
    parser.add_argument('--cache_only', action='store_true')
    args = parser.parse_args()

    if not args.target_task:
        args.target_task = ['TCGA-LUNG-subtype']
    if not args.control_task:
        args.control_task = ['UBC-OCEAN-subtype', 'BACH-histology']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_tasks = list(dict.fromkeys(args.target_task + args.control_task))
    datasets = list(dict.fromkeys(get_task(t)['dataset'] for t in all_tasks))
    organ_of = {}
    meta = pd.read_csv(args.metadata)
    for ds in datasets:
        organ_of[ds] = select_dataset(meta, ds)['organ'].iloc[0]

    def enc_dir(ds):
        root = (f"{args.base_dir}/trident_features/master_benchmark"
                if ds.startswith('TCGA') else f"{args.base_dir}/{ds}")
        return f"{root}/20x_224px_0px_overlap/{args.fm}"

    print(f"Quick validation | fm={args.fm} device={device}")
    print(f"targets : {args.target_task}")
    print(f"controls: {args.control_task}\n")
    print("Loading mean-pooled features (one load per dataset)...", flush=True)

    dataset_cache = {}
    for ds in datasets:
        dataset_cache[ds] = load_pooled(
            ds, enc_dir(ds), args.metadata, args.patches, args.workers,
            args.cache_dir, args.seed, args.block, args.max_slides,
        )

    if args.cache_only:
        print("\nCache built; exiting (--cache_only).")
        return

    tasks = {t: build_task(t, dataset_cache, args.metadata, args.fold) for t in all_tasks}
    d = next(iter(tasks.values()))[0].shape[1]

    print(f"\nTask sizes (fold {args.fold}, d={d}):")
    for t, (Xtr, ytr, Xte, yte, nc) in tasks.items():
        print(f"  {t:22s} {get_task(t)['dataset']:12s} train {len(Xtr):5d} / "
              f"test {len(Xte):5d}  ({nc} classes)")

    def np_transform(module):
        def fn(X):
            with torch.no_grad():
                return module(torch.tensor(X, device=device)).cpu().numpy()
        return fn

    print("\n=== baseline (no erasure) ===", flush=True)
    baseline = evaluate(lambda X: X, tasks, args.probes)
    show(baseline, all_tasks)

    matrix = {'baseline': baseline}

    for target in args.target_task:
        Xtr_t, ytr_t = tasks[target][0], tasks[target][1]
        controls = [c for c in args.control_task if c != target]
        control_train = {c: (tasks[c][0], tasks[c][1]) for c in controls}
        order = [target] + controls

        if 'svd' in args.methods:
            for k in args.svd_ranks:
                tag = f"{target}|svd_k{k}"
                print(f"\n=== [{target}] SVD subspace removal (k={k}) ===", flush=True)
                U = svd_subspace(torch.tensor(Xtr_t), k=k).to(device)
                mu = torch.tensor(Xtr_t, device=device).mean(0)
                if args.no_svd_affine:
                    fn = lambda X, U=U: (lambda t: (t - (t @ U) @ U.T).cpu().numpy())(
                        torch.tensor(X, device=device))
                else:
                    fn = lambda X, U=U, mu=mu: remove_subspace_affine(
                        torch.tensor(X, device=device), U, mu).cpu().numpy()
                matrix[tag] = evaluate(fn, tasks, args.probes)
                show(matrix[tag], order)

        def _apply(U_, tag_, order_):
            U_ = U_.to(device)
            mu_ = torch.tensor(Xtr_t, device=device).mean(0)
            fn_ = lambda X, U_=U_, mu_=mu_: remove_subspace_affine(
                torch.tensor(X, device=device), U_, mu_).cpu().numpy()
            matrix[tag_] = evaluate(fn_, tasks, args.probes)
            show(matrix[tag_], order_)

        def _apply(U_, tag_, order_):
            U_ = U_.to(device)
            mu_ = torch.tensor(Xtr_t, device=device).mean(0)
            fn_ = lambda X, U_=U_, mu_=mu_: remove_subspace_affine(
                torch.tensor(X, device=device), U_, mu_).cpu().numpy()
            matrix[tag_] = evaluate(fn_, tasks, args.probes)
            show(matrix[tag_], order_)

        if 'spectral' in args.methods:
            for lam in args.spectral_lam:
                for k in args.svd_ranks:
                    tag = f"{target}|spectral_l{lam}_k{k}"
                    print(f"\n=== [{target}] spectral pencil "
                          f"(lambda={lam}, k={k}) ===", flush=True)
                    U, diag = spectral_subspace(
                        torch.tensor(Xtr_t), k,
                        controls=[torch.tensor(tasks[c][0]) for c in controls],
                        lam=lam, return_diagnostics=True)
                    print(f"    target variance captured "
                          f"{diag['target_variance_captured']:.4f}")
                    _apply(U, tag, order)

        if 'leace' in args.methods:
            tag = f"{target}|leace"
            print(f"\n=== [{target}] LEACE (closed form, erases the target label) ===", flush=True)
            try:
                from src.unlearning.concept_erasure import LeaceFitter
                z_oh = torch.nn.functional.one_hot(
                    torch.tensor(ytr_t, dtype=torch.long), num_classes=tasks[target][4]
                ).double()
                er = LeaceFitter.fit(torch.tensor(Xtr_t).double(), z_oh).eraser
                P = er.P.float().to(device)
                mu = er.bias.float().to(device)
                fn = lambda X: (lambda t: ((t - mu) @ P.T + mu).cpu().numpy())(
                    torch.tensor(X, device=device))
                matrix[tag] = evaluate(fn, tasks, args.probes)
                show(matrix[tag], order)
            except Exception as e:
                print(f"  LEACE failed: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------ #
    for probe in args.probes:
        print("\n" + "=" * 118)
        print(f"SUMMARY  probe={probe} (retrained from scratch; chance = 0.50)")
        print("=" * 118)
        for target in args.target_task:
            controls = [c for c in args.control_task if c != target]
            print(f"\nTARGET: {target}  ({get_task(target)['description']})")
            hdr = f"  {'method':<22}{'TARGET':>10}"
            for c in controls:
                hdr += f"{c[:14]:>16}"
            hdr += f"{'tgt drop':>10}{'collat':>9}{'SDS':>7}"
            print(hdr)
            rels = "  " + " " * 22 + " " * 10 + "".join(
                f"{relation(target, c, organ_of)[:14]:>16}" for c in controls)
            print(rels)
            print("  " + "-" * 114)

            for tag, res in matrix.items():
                if tag != 'baseline' and not tag.startswith(target + '|'):
                    continue
                label = 'baseline' if tag == 'baseline' else tag.split('|', 1)[1]
                tgt = res[target].get(probe)
                row = f"  {label:<22}" + (f"{tgt:>10.4f}" if tgt is not None else f"{'n/a':>10}")
                drops = []
                for c in controls:
                    v = res[c].get(probe)
                    row += f"{v:>16.4f}" if v is not None else f"{'n/a':>16}"
                    b = baseline[c].get(probe)
                    if v is not None and b is not None:
                        drops.append(b - v)
                bt = baseline[target].get(probe)
                if tag == 'baseline' or tgt is None or bt is None:
                    print(row)
                    continue
                tdrop = bt - tgt
                coll = float(np.mean(drops)) if drops else float('nan')
                sds = selective_degradation_score(tdrop, drops) if drops else float('nan')
                print(row + f"{tdrop:>+10.4f}{coll:>+9.4f}{sds:>7.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'args': vars(args), 'organ_of': organ_of,
                   'n_classes': {t: tasks[t][4] for t in tasks},
                   'results': matrix}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
