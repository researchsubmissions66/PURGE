"""
Collect sweep results into tidy tables, one section per ablation axis.

Reads results/sweep_v2/*.json (one per config) and writes:

    results/sweep_v2/_tidy.csv      one row per (run, target, control, probe)
    results/sweep_v2/_summary.csv   one row per (run, target, probe)
    stdout                          a table per axis

Chance for macro one-vs-rest AUC is 0.50 for any number of classes.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

AXIS_KEY = {'encoder': 'fm', 'fold': 'fold', 'rank': 'k',
            'method': 'method', 'spectral': 'spectral_lam', 'base': 'method'}


def load(out_dir):
    rows, summary, failed = [], [], []
    for f in sorted(glob.glob(os.path.join(out_dir, '*.json'))):
        if os.path.basename(f).startswith('_'):
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        cfg = r.get('config', {})
        if r.get('status') != 'ok':
            failed.append({'run_id': cfg.get('run_id', os.path.basename(f)),
                           'axis': cfg.get('axis'), 'error': r.get('error')})
            continue
        for target, m in r.get('metrics', {}).items():
            for probe in cfg.get('probes', []):
                pm = m.get(probe)
                if not pm:
                    continue
                summary.append({
                    'run_id': cfg['run_id'], 'axis': cfg['axis'],
                    'fm': cfg['fm'], 'fold': cfg['fold'], 'method': cfg['method'],
                    'k': cfg['k'], 'spectral_lam': cfg.get('spectral_lam'),
                    'target': target, 'probe': probe,
                    'baseline': pm['baseline'], 'erased': pm['erased'],
                    'target_drop': pm['target_drop'],
                    'collateral': pm['collateral_mean'], 'sds': pm['sds'],
                    'centred_cosine': m.get('centred_cosine'),
                    'seconds': r.get('seconds'),
                })
                for ctrl, cm in pm['per_control'].items():
                    rows.append({
                        'run_id': cfg['run_id'], 'axis': cfg['axis'],
                        'fm': cfg['fm'], 'fold': cfg['fold'],
                        'method': cfg['method'], 'k': cfg['k'],
                        'target': target, 'control': ctrl,
                        'relation': cm['relation'], 'probe': probe,
                        'control_baseline': cm['baseline'],
                        'control_erased': cm['erased'],
                        'control_drop': cm['baseline'] - cm['erased'],
                    })
    return pd.DataFrame(rows), pd.DataFrame(summary), pd.DataFrame(failed)


def fmt(df, axis_col, probe):
    d = df[df.probe == probe]
    if d.empty:
        return None
    piv = d.pivot_table(index=axis_col, columns='target',
                        values='erased', aggfunc='mean')
    coll = d.pivot_table(index=axis_col, columns='target',
                         values='collateral', aggfunc='mean')
    return piv, coll


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out_dir', default='results/sweep_v2')
    ap.add_argument('--probe', default='mlp')
    args = ap.parse_args()

    tidy, summ, failed = load(args.out_dir)
    if summ.empty:
        raise SystemExit(f"No completed results in {args.out_dir}")

    tidy.to_csv(os.path.join(args.out_dir, '_tidy.csv'), index=False)
    summ.to_csv(os.path.join(args.out_dir, '_summary.csv'), index=False)

    n_cfg = len(glob.glob('configs/sweeps/*.json'))
    print(f"{summ.run_id.nunique()}/{n_cfg} configs complete"
          f"{f', {len(failed)} FAILED' if len(failed) else ''}")
    if len(failed):
        for _, r in failed.iterrows():
            print(f"  FAILED {r.run_id}: {r.error}")
    print(f"\nprobe = {args.probe}   (chance = 0.50)\n")

    for axis in ['encoder', 'fold', 'rank', 'method', 'spectral']:
        d = summ[summ.axis == axis]
        if d.empty:
            continue
        col = AXIS_KEY[axis]
        out = fmt(d, col, args.probe)
        if out is None:
            continue
        erased, coll = out
        print("=" * 78)
        print(f"AXIS: {axis}   (rows = {col})")
        print("=" * 78)
        print("\n-- target AUC after erasure (lower = stronger) --")
        print(erased.round(4).to_string())
        print("\n-- mean collateral on controls (lower = better) --")
        print(coll.round(4).to_string())
        if axis == 'fold':
            g = d.groupby('target').erased.agg(['mean', 'std', 'count'])
            print("\n-- across folds --")
            print(g.round(4).to_string())
        print()

    same = tidy[(tidy.probe == args.probe) & (tidy.relation == 'same-slides')]
    if not same.empty:
        print("=" * 78)
        print("SAME-SLIDES CONTROLS  (the unconfounded comparison)")
        print("=" * 78)
        g = same.groupby(['axis', 'target', 'control']).agg(
            control_baseline=('control_baseline', 'mean'),
            control_erased=('control_erased', 'mean'),
            drop=('control_drop', 'mean'), n=('control_drop', 'size'))
        print(g.round(4).to_string())

    print(f"\nWrote {args.out_dir}/_tidy.csv and _summary.csv")


if __name__ == "__main__":
    main()
