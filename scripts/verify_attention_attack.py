"""
Cheap check of the mechanism before spending 45 MIL runs.

CLAIM: attention resists plain SVD because label signal survives in the
WITHIN-SLIDE DEVIATIONS z_j - mean(z), which mean pooling averages away but
selection can exploit. If true then:
  - a probe on the slide MEAN collapses under plain svd   (MeanMIL proxy)
  - a probe on [mean | max | std] does NOT                (attention proxy)
  - removing the deviation subspace too should collapse both
No MIL training: [mean|max|std] is a permutation-invariant bag summary that
upper-bounds what a weighted average of patches can read, at ~1000x less compute.
"""
import sys, os, numpy as np, torch, h5py, pandas as pd, collections
sys.path.insert(0, '/u/dchanda/PURGE')
from scripts.fit_unlearner import build_index
from src.datasets.feature_dataset import to_h5_name
from src.unlearning.subspace import svd_subspace, remove_subspace_affine, orthonormalize
from src.utils.splits import patient_folds
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DS='BRACS'; K=64; KDEV=64; NSLIDES=700; PATCH=48; FOLD=0
ENC='/work/hdd/bhwm/BRACS/20x_224px_0px_overlap/features_virchow2'
md = pd.read_csv('data/multi_benchmark_metadata.csv')
sub = md[(md.dataset==DS) & (md.label.isin(['ADH','FEA']))]      # the atypia task
print(f"{DS} atypia slides available: {len(sub)}")
if len(sub) > NSLIDES: sub = sub.sample(NSLIDES, random_state=0)
idx = build_index([ENC])

bags, ys, pats = [], [], []
for _, r in sub.iterrows():
    p = idx.get(to_h5_name(r['filename']))
    if p is None: continue
    try:
        with h5py.File(p,'r') as f:
            z = f['features'][:]
    except Exception: continue
    if z.shape[0] < 4: continue
    if z.shape[0] > PATCH:
        s = np.linspace(0, z.shape[0]-1, PATCH).astype(int)
        z = z[s]
    bags.append(torch.tensor(np.asarray(z, np.float32)))
    ys.append(1 if r['label']=='FEA' else 0); pats.append(r['patient_id'])
ys=np.array(ys); pats=np.array(pats)
print(f"read {len(bags)} bags, mean {np.mean([b.shape[0] for b in bags]):.0f} patches, "
      f"class balance {dict(collections.Counter(ys))}")

tr_p, te_p = patient_folds('data/multi_benchmark_metadata.csv', DS, FOLD)
tr = np.isin(pats, tr_p); te = np.isin(pats, te_p)
print(f"fold {FOLD}: train {tr.sum()} test {te.sum()}")

means = torch.stack([b.mean(0) for b in bags])
mu = means[tr].mean(0)
U  = svd_subspace(means[tr], k=K)                                  # the mean subspace
devs = torch.cat([bags[i] - bags[i].mean(0, keepdim=True) for i in np.where(tr)[0]])
# randomized SVD: a full decomposition of a 13k x 2560 matrix on a loaded login
# node is minutes; this is seconds and the top-64 directions are identical to
# several decimals.
_, _, Vd = torch.pca_lowrank(devs, q=KDEV + 16, center=True, niter=4)
Ud = Vd[:, :KDEV].contiguous()                                     # deviation subspace
print(f"U {tuple(U.shape)}  U_dev {tuple(Ud.shape)}  pooled deviations {tuple(devs.shape)}")
overlap = float(torch.linalg.svdvals(U.double().T @ Ud.double()).mean())
print(f"mean/deviation subspace overlap: {overlap:.3f}  (low => they are different directions)")
Uboth = orthonormalize(torch.cat([U, Ud], 1))

def summarise(transform):
    M, X = [], []
    for b in bags:
        e = transform(b)
        M.append(e.mean(0))
        X.append(torch.cat([e.mean(0), e.max(0).values, e.std(0)]))
    return torch.stack(M).numpy(), torch.stack(X).numpy()

def auc(A):
    m = LogisticRegression(max_iter=3000).fit(A[tr], ys[tr])
    a = roc_auc_score(ys[te], m.predict_proba(A[te])[:,1])
    return max(a, 1-a)

Z = torch.zeros_like(mu)
ARMS = [
 ('none                ', lambda b: b),
 ('svd  (mean subspace)', lambda b: remove_subspace_affine(b, U, mu)),
 ('svd_dev (mean+dev)  ', lambda b: remove_subspace_affine(b, Uboth, mu)),
 ('svd_bag alpha=0.5   ', lambda b: remove_subspace_affine(b.mean(0,keepdim=True), U, mu)
                                   + 0.5*remove_subspace_affine(b-b.mean(0,keepdim=True), U, Z)),
 ('svd_bag alpha=0     ', lambda b: remove_subspace_affine(b.mean(0,keepdim=True), U, mu)
                                   .expand_as(b)),
]
print(f"\n{'arm':22s} {'mean-only (MeanMIL proxy)':>26s} {'mean|max|std (attn proxy)':>27s}")
for name, t in ARMS:
    M, X = summarise(t)
    print(f"{name:22s} {auc(M):>26.4f} {auc(X):>27.4f}")
print("\nchance = 0.50.  If the attention proxy stays high under plain svd and")
print("falls under svd_dev, the mechanism holds and the full MIL runs are worth it.")
