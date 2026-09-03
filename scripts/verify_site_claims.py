"""
Check every numeric claim on docs/index.html against the data and result files.

Run this before publishing the site. It caught two errors: an encoder table that
listed a model never run (Phikon-v2), omitted four that were, and had two
dimensions wrong; and a patch count understated by ~16x.

Note on the task count: the number of TASKS is len(result["baseline"]) - three
targets plus four controls. result["metrics"] holds only the targets, and
counting that instead reports 3 and looks like a site error when it is not.
"""
import json, glob, os, re, sys, collections
import numpy as np, pandas as pd
sys.path.insert(0, '/u/dchanda/PURGE')

ok = lambda c: "OK   " if c else "WRONG"
rows = []
def check(claim, actual, passes, note=""):
    rows.append((ok(passes), claim, actual, note))

md = pd.read_csv('data/multi_benchmark_metadata.csv')
check("18K+ whole slide images", f"{len(md)} rows in metadata", len(md) >= 18000)
check("5 benchmark datasets", f"{sorted(md.dataset.unique())}", md.dataset.nunique() == 5)
check("4 target organs", f"{sorted(md.organ.dropna().unique())}", md.organ.nunique() == 4)

cfgs = glob.glob('configs/sweeps_full/*.json')
res  = [f for f in glob.glob('results/sweep_full/*.json') if '/_' not in f]
check("416 sweep configurations", f"{len(cfgs)} configs, {len(res)} results", len(cfgs) == 416)

status = collections.Counter(json.load(open(f)).get('status') for f in res)
check("zero failed runs", dict(status), status.get('failed', 0) == 0)

fms, dims, folds = set(), {}, collections.defaultdict(set)
tasks = set()
for f in res:
    r = json.load(open(f)); c = r['config']
    fms.add(c['fm']); folds[c['fm']].add(c['fold'])
    if r.get('input_dim'): dims[c['fm']] = r['input_dim']
    tasks |= set(r.get('baseline', {}))   # targets AND controls
check("8 foundation models", sorted(x.replace('features_','') for x in fms), len(fms) == 8)
check("7 downstream tasks", sorted(tasks), len(tasks) == 7)
check("5 folds on every encoder", {k.replace('features_',''): sorted(v) for k,v in folds.items()},
      all(v == {0,1,2,3,4} for v in folds.values()))

SITE_DIMS = {'virchow2':2560,'virchow':2560,'hoptimus0':1536,'uni_v2':1536,
             'gigapath':1536,'gpfm':1024,'conch_v15':768,'ctranspath':768}
bad = {k: (SITE_DIMS[k], dims.get('features_'+k)) for k in SITE_DIMS
       if dims.get('features_'+k) != SITE_DIMS[k]}
check("encoder dimensions on site", bad or "all match", not bad)

# ---- patch count: sample shapes per cohort, scale by cohort size ---------- #
import h5py
from src.datasets.feature_dataset import to_h5_name
ROOT = '/work/hdd/bhwm'; V = '20x_224px_0px_overlap/features_virchow2'
DIRS = {'TCGA': f'{ROOT}/trident_features/master_benchmark/{V}', 'PANDA': f'{ROOT}/PANDA/{V}',
        'BRACS': f'{ROOT}/BRACS/{V}', 'BACH': f'{ROOT}/BACH/{V}', 'UBC-OCEAN': f'{ROOT}/UBC-OCEAN/{V}'}
rng = np.random.RandomState(0); total_est = 0; detail = {}
for ds, d in DIRS.items():
    sub = md[md.dataset == ds]
    if sub.empty or not os.path.isdir(d): continue
    samp = sub.sample(min(40, len(sub)), random_state=0)
    counts = []
    for _, r in samp.iterrows():
        try:
            with h5py.File(os.path.join(d, to_h5_name(r['filename'])), 'r') as h:
                counts.append(h['features'].shape[0])
        except Exception:
            pass
    if not counts: continue
    mean = float(np.mean(counts))
    est = mean * len(sub)
    detail[ds] = f"{len(sub)} slides x {mean:.0f} patches = {est/1e6:.2f}M"
    total_est += est
check("3.5M+ processed patches", f"{total_est/1e6:.2f}M estimated  |  " + "; ".join(f"{k}: {v}" for k,v in detail.items()),
      total_est >= 3.5e6, "sampled 40 slides/cohort, virchow2 @224px")

w = max(len(r[1]) for r in rows)
print(f"{'':5} {'CLAIM':{w}}  ACTUAL")
print("-" * (w + 60))
for st, claim, actual, note in rows:
    print(f"{st} {claim:{w}}  {actual}" + (f"   [{note}]" if note else ""))
