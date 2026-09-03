"""Collect the MIL confirmation sweep into a target x control matrix per model."""
import argparse, glob, json, os, re
import numpy as np, pandas as pd

BASE = re.compile(r'^base_(?P<ds>.+)_(?P<model>abmil|transmil|meanmil)_f(?P<fold>\d+)\.json$')
# erase-ORGAN_* was the old organ-level patch-fitted eraser (quarantined).
# erase_* is the current per-(dataset, fold) slide-mean eraser.
ERAS = re.compile(r'^erase[_-](?P<organ>[A-Z]+_)?(?P<ds>.+)_(?P<model>abmil|transmil|meanmil)_f(?P<fold>\d+)\.json$')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out_dir', default='results/mil_sweep')
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.out_dir, '*.json'))):
        n = os.path.basename(f)
        try:
            d = json.load(open(f))
        except Exception:
            continue
        auc = d.get('test_auc')
        if auc is None:
            continue
        m = BASE.match(n) or ERAS.match(n)
        if not m:
            continue
        g = m.groupdict()
        rows.append(dict(dataset=g['ds'], model=g['model'], fold=int(g['fold']),
                         organ=(g.get('organ') or 'SELF').rstrip('_'), auc=auc,
                         val_auc=d.get('val_auc'), n_test=d.get('n_test')))
    if not rows:
        raise SystemExit(f"no MIL results in {a.out_dir}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out_dir, '_mil.csv'), index=False)
    print(f"{len(df)} runs | chance = 0.50\n")
    for model, sub in df.groupby('model'):
        print("=" * 72)
        print(f"MODEL: {model}   (rows = erased organ, cols = evaluated dataset)")
        print("=" * 72)
        piv = sub.pivot_table(index='organ', columns='dataset', values='auc',
                              aggfunc='mean')
        sd = sub.pivot_table(index='organ', columns='dataset', values='auc',
                             aggfunc='std')
        print(piv.round(4).to_string())
        print("\n-- std over folds --")
        print(sd.round(4).to_string())
        print()


if __name__ == '__main__':
    main()
