"""
Run ONE sweep config and write ONE result file.

Idempotent: if the result exists and is complete, it is skipped, so the sweep is
resumable and re-launchable without bookkeeping. Every result embeds the config
that produced it, so a result file is self-describing.

Erasers are fitted on the TARGET TRAIN split only and evaluated with probes
retrained from scratch, per the protocol in AGENTS.md.
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)

from quick_validate import build_task, evaluate, load_pooled          # noqa: E402
from src.evaluation.metrics import selective_degradation_score        # noqa: E402
from src.evaluation.tasks import get_task, relation                   # noqa: E402
from src.unlearning.noise import apply_dropout, apply_gaussian_noise  # noqa: E402
from src.unlearning.spectral import spectral_subspace                 # noqa: E402
from src.unlearning.subspace import (remove_subspace,                 # noqa: E402
                                     remove_subspace_affine, svd_subspace)


def encoder_dir(fm, dataset, base_dir='/work/hdd/bhwm'):
    root = (f"{base_dir}/trident_features/master_benchmark"
            if dataset.startswith('TCGA') else f"{base_dir}/{dataset}")
    if fm == 'features_conch_v15' and dataset.startswith('TCGA'):
        px = 512
    elif fm in ('features_uni_v2', 'features_gigapath',
                'features_ctranspath', 'features_conch_v15'):
        px = 256
    else:
        px = 224
    return f"{root}/20x_{px}px_0px_overlap/{fm}"


def subsample_fit(Xtr, ytr, n_fit, seed):
    """
    Restrict what the ATTACKER sees when fitting the eraser.

    Threat-model parameter, not an implementation detail: the attacker's sample
    budget determines whether the attack is practical. Fitting on 2000 labelled
    WSIs is a very different claim from fitting on 50.
    """
    if not n_fit or n_fit >= len(Xtr):
        return Xtr, ytr
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Xtr), n_fit, replace=False)
    return Xtr[idx], ytr[idx]


def build_transform(cfg, Xtr, ytr, control_X, device, n_classes):
    """
    Return a numpy->numpy transform fitted on the TARGET TRAIN split only.

    The transform is then applied to train AND test of every task: the victim
    never sees clean features. That is the poisoning threat model (poisoned
    dataset or poisoned encoder), not evasion. Fitting on train and evaluating on
    held-out test keeps the attacker away from the victim's evaluation data.
    """
    Xtr, ytr = subsample_fit(Xtr, ytr, cfg.get('n_fit'), cfg.get('seed', 0))
    build_transform.effective_k = cfg['k']
    build_transform.n_fit_used = len(Xtr)
    method, k = cfg['method'], cfg['k']
    Xt = torch.tensor(Xtr)
    mu = Xt.mean(0).to(device)

    if method in ('svd', 'svd_plain'):
        U = svd_subspace(Xt, k=k).to(device)
        build_transform.effective_k = int(U.shape[1])
        if method == 'svd':
            return lambda X: remove_subspace_affine(
                torch.tensor(X, device=device), U, mu).cpu().numpy()
        return lambda X: remove_subspace(
            torch.tensor(X, device=device), U).cpu().numpy()

    if method == 'spectral':
        U = spectral_subspace(Xt, k, controls=[torch.tensor(C) for C in control_X],
                              lam=cfg.get('spectral_lam', 1.0)).to(device)
        build_transform.effective_k = int(U.shape[1])
        return lambda X: remove_subspace_affine(
            torch.tensor(X, device=device), U, mu).cpu().numpy()

    if method == 'leace':
        from src.unlearning.concept_erasure import LeaceFitter
        z_oh = torch.nn.functional.one_hot(
            torch.tensor(ytr, dtype=torch.long), num_classes=n_classes).double()
        er = LeaceFitter.fit(Xt.double(), z_oh).eraser
        P, b = er.P.float().to(device), er.bias.float().to(device)
        return lambda X: (lambda t: ((t - b) @ P.T + b).cpu().numpy())(
            torch.tensor(X, device=device))

    if method == 'splince':
        from src.unlearning.splince.proj import proj
        p = proj()
        p.fit(Xt.double().numpy(), ytr.reshape(-1, 1),
              np.zeros((len(ytr), 1)), method='SPLINCE')
        P = torch.tensor(p.P, dtype=torch.float32, device=device)
        b = torch.tensor(p.b, dtype=torch.float32, device=device)
        return lambda X: ((torch.tensor(X, device=device) @ P.T) + b).cpu().numpy()

    if method == 'low_rank':
        # NEGATIVE CONTROL. Invertible by construction, so it must NOT erase.
        from src.unlearning.low_rank import LowRankEraser
        er = LowRankEraser(input_dim=Xtr.shape[1], rank=k).to(device)
        with torch.no_grad():
            er.B.weight.normal_(0, 0.05)
        er.eval()
        return lambda X: er(torch.tensor(X, device=device)).detach().cpu().numpy()

    if method == 'gaussian':
        s = cfg.get('sigma', 1.0)
        return lambda X: apply_gaussian_noise(torch.tensor(X), sigma=s).numpy()

    if method == 'dropout':
        p_ = cfg.get('dropout_p', 0.5)
        return lambda X: apply_dropout(torch.tensor(X), p=p_).numpy()

    raise ValueError(f"unknown method {method!r}")


def centred_cosine(A, B):
    a = A - A.mean(0, keepdims=True)
    b = B - B.mean(0, keepdims=True)
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return float(np.mean(num / np.maximum(den, 1e-12)))


def run(cfg, metadata, cache_dir, out_path, device):
    t0 = time.time()
    all_tasks = list(dict.fromkeys(cfg['targets'] + cfg['controls']))
    datasets = list(dict.fromkeys(get_task(t)['dataset'] for t in all_tasks))

    # Feature coverage is not uniform: conch_v15 and virchow have ZERO BACH slides
    # while every other encoder has 400. Skip the affected tasks and record it,
    # rather than voiding the whole config over one missing cohort.
    cache, unavailable = {}, []
    for ds in datasets:
        try:
            cache[ds] = load_pooled(ds, encoder_dir(cfg['fm'], ds), metadata,
                                    cfg['patches'], 16, cache_dir, cfg['seed'],
                                    32, cfg['max_slides'])
        except SystemExit as e:
            print(f"  UNAVAILABLE {ds} for {cfg['fm']}: {e}")
            unavailable.append(ds)

    all_tasks = [t for t in all_tasks if get_task(t)['dataset'] in cache]
    dropped = [t for t in (cfg['targets'] + cfg['controls'])
               if get_task(t)['dataset'] not in cache]
    if not all_tasks:
        raise SystemExit(f"no usable tasks for {cfg['fm']}")

    tasks = {t: build_task(t, cache, metadata, cfg['fold'],
                           n_splits=cfg.get('n_splits', 5)) for t in all_tasks}
    d = next(iter(tasks.values()))[0].shape[1]

    import pandas as pd
    meta = pd.read_csv(metadata)
    organ_of = {}
    from src.datasets.feature_dataset import select_dataset
    for ds in datasets:
        organ_of[ds] = select_dataset(meta, ds)['organ'].iloc[0]

    baseline = evaluate(lambda X: X, tasks, cfg['probes'])
    result = {'config': cfg, 'input_dim': d, 'baseline': baseline,
              'erased': {}, 'metrics': {},
              'n': {t: {'train': len(tasks[t][0]), 'test': len(tasks[t][2]),
                        'classes': tasks[t][4]} for t in tasks}}

    for target in [t for t in cfg['targets'] if t in tasks]:
        # fit_on lets the eraser be fitted on a DIFFERENT cohort than the one
        # attacked - the poisoned-encoder story, where the victim's data was never
        # seen by the attacker.
        fit_task = cfg.get('fit_on') or target
        if fit_task not in tasks:
            fit_task = target
        Xtr, ytr = tasks[fit_task][0], tasks[fit_task][1]
        controls = [c for c in cfg['controls'] if c != target and c in tasks]
        fn = build_transform(cfg, Xtr, ytr, [tasks[c][0] for c in controls],
                             device, tasks[fit_task][4])
        res = evaluate(fn, tasks, cfg['probes'])
        result['erased'][target] = res
        result.setdefault('fit_on', {})[target] = fit_task
        result.setdefault('n_fit_used', {})[target] = getattr(
            build_transform, 'n_fit_used', len(tasks[fit_task][0]))
        result.setdefault('effective_k', {})[target] = getattr(
            build_transform, 'effective_k', cfg['k'])

        m = {}
        for probe in cfg['probes']:
            b, a = baseline[target].get(probe), res[target].get(probe)
            drops = [baseline[c][probe] - res[c][probe] for c in controls
                     if baseline[c].get(probe) is not None
                     and res[c].get(probe) is not None]
            if b is None or a is None:
                continue
            m[probe] = {
                'baseline': b, 'erased': a, 'target_drop': b - a,
                'collateral_mean': float(np.mean(drops)) if drops else None,
                'sds': selective_degradation_score(b - a, drops) if drops else None,
                'per_control': {c: {'baseline': baseline[c][probe],
                                    'erased': res[c][probe],
                                    'relation': relation(target, c, organ_of)}
                                for c in controls},
            }
        Xte = tasks[target][2]
        m['centred_cosine'] = centred_cosine(Xte, fn(Xte))
        result['metrics'][target] = m

    result['unavailable_datasets'] = unavailable
    result['dropped_tasks'] = sorted(set(dropped))
    result['status'] = 'ok'
    result['seconds'] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = f"{out_path}.tmp.{os.getpid()}"
    json.dump(result, open(tmp, 'w'), indent=2)
    os.replace(tmp, out_path)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--metadata', default='data/multi_benchmark_metadata.csv')
    ap.add_argument('--cache_dir', default='results/quick/cache')
    ap.add_argument('--out_dir', default='results/sweep_v2')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--cache_only', action='store_true',
                    help="build feature caches for this config's encoder, then exit")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    out_path = os.path.join(args.out_dir, f"{cfg['run_id']}.json")

    if args.cache_only:
        all_tasks = list(dict.fromkeys(cfg['targets'] + cfg['controls']))
        for ds in dict.fromkeys(get_task(t)['dataset'] for t in all_tasks):
            load_pooled(ds, encoder_dir(cfg['fm'], ds), args.metadata,
                        cfg['patches'], 16, args.cache_dir, cfg['seed'],
                        32, cfg['max_slides'])
        print(f"caches warm for {cfg['fm']}")
        return

    # Reuse an existing result ONLY if it was produced by an identical config.
    # run_id is derived from the VARIED keys only, so editing a base setting
    # (max_slides 600 -> 2000, n_splits, the control list) leaves the run_id
    # unchanged while the meaning changes. Without this check those results are
    # silently reused and the analysis pools incomparable settings - 17 files were
    # contaminated that way before it was caught.
    SETTINGS_KEYS = ('fm', 'fold', 'n_splits', 'method', 'k', 'max_slides',
                     'patches', 'seed', 'spectral_lam', 'sigma', 'dropout_p',
                     'n_fit', 'fit_on', 'targets', 'controls', 'probes')
    if os.path.exists(out_path) and not args.force:
        try:
            prev = json.load(open(out_path))
            prev_cfg = prev.get('config', {})
            differs = [k for k in SETTINGS_KEYS if prev_cfg.get(k) != cfg.get(k)]
            if prev.get('status') == 'ok' and not differs:
                print(f"SKIP {cfg['run_id']} (already complete)")
                return
            if differs:
                print(f"STALE {cfg['run_id']}: recomputing, differs in {differs}")
        except Exception:
            pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    print(f"RUN {cfg['run_id']}  [axis={cfg['axis']}]  device={device}", flush=True)

    try:
        r = run(cfg, args.metadata, args.cache_dir, out_path, device)
        for t, m in r['metrics'].items():
            mlp = m.get('mlp', {})
            if mlp:
                print(f"  {t:22s} {mlp['baseline']:.4f} -> {mlp['erased']:.4f}"
                      f"  drop {mlp['target_drop']:+.4f}"
                      f"  collat {mlp['collateral_mean']:+.4f}"
                      if mlp.get('collateral_mean') is not None else "")
        print(f"OK {cfg['run_id']} in {r['seconds']}s -> {out_path}")
    except Exception as e:
        os.makedirs(args.out_dir, exist_ok=True)
        json.dump({'config': cfg, 'status': 'failed',
                   'error': f"{type(e).__name__}: {e}",
                   'traceback': traceback.format_exc()},
                  open(out_path, 'w'), indent=2)
        print(f"FAILED {cfg['run_id']}: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
