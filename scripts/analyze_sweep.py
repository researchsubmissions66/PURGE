"""
Full analysis of the PURGE sweep.

Reads results/sweep_v2/*.json and produces every analysis the design supports:

  1  main effects per axis, with fold error bars
  2  encoder x fold interaction - is encoder-dependence real or fold noise?
  3  significance: is erasure distinguishable from baseline, and from chance?
  4  saturation: where does k stop helping, per encoder
  5  confound ladder: same-slides vs same-organ vs cross-organ collateral
  6  negative-control validation (low_rank must NOT erase)
  7  superset check (spectral lam=0 must equal svd)
  8  what predicts erasability - baseline AUC, embedding dim, sample size
  9  best and worst configurations
 10  variance decomposition: how much of the spread is fold vs encoder

Chance for macro one-vs-rest AUC is 0.50 for any class count.
"""

import argparse
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

# the probes the base config runs; the probe_family axis is analysed separately
BASE_PROBES = ('logreg', 'mlp')

warnings.filterwarnings('ignore')


def load(out_dir):
    rows, summ, failed = [], [], []
    for f in sorted(glob.glob(os.path.join(out_dir, '*.json'))):
        if os.path.basename(f).startswith('_'):
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        cfg = r.get('config', {})
        if r.get('status') != 'ok':
            failed.append({'run_id': cfg.get('run_id'), 'error': r.get('error')})
            continue
        dim = r.get('input_dim')
        for target, m in r.get('metrics', {}).items():
            for probe in cfg.get('probes', []):
                pm = m.get(probe)
                if not pm:
                    continue
                summ.append(dict(
                    run_id=cfg['run_id'], axis=cfg['axis'], fm=cfg['fm'],
                    fold=cfg['fold'], method=cfg['method'], k=cfg['k'],
                    spectral_lam=cfg.get('spectral_lam'), sigma=cfg.get('sigma'),
                    dropout_p=cfg.get('dropout_p'), patches=cfg.get('patches'),
                    max_slides=cfg.get('max_slides'), seed=cfg.get('seed'),
                    target=target, probe=probe, dim=dim,
                    baseline=pm['baseline'], erased=pm['erased'],
                    target_drop=pm['target_drop'],
                    collateral=pm['collateral_mean'], sds=pm['sds'],
                    centred_cosine=m.get('centred_cosine'),
                    n_train=r['n'][target]['train'], n_test=r['n'][target]['test'],
                    n_fit=cfg.get('n_fit'),
                    n_fit_used=(r.get('n_fit_used') or {}).get(target),
                    effective_k=(r.get('effective_k') or {}).get(target),
                    fit_on=(r.get('fit_on') or {}).get(target),
                ))
                for c, cm in pm['per_control'].items():
                    rows.append(dict(
                        run_id=cfg['run_id'], axis=cfg['axis'], fm=cfg['fm'],
                        fold=cfg['fold'], method=cfg['method'], k=cfg['k'],
                        target=target, control=c, relation=cm['relation'],
                        probe=probe, control_baseline=cm['baseline'],
                        control_erased=cm['erased'],
                        control_drop=cm['baseline'] - cm['erased']))
    return pd.DataFrame(rows), pd.DataFrame(summ), pd.DataFrame(failed)


def base_settings(df):
    """
    Restrict to base configuration, varying nothing.

    Without this, filtering on (method, k, fm) alone sweeps in the budget and
    sample_size axes - configs with different n_fit and max_slides get pooled and
    reported as "fold variance", which massively inflates it and breaks like-for-
    like comparisons (the spectral lam=0 == svd check failed for exactly this).

    fit_on and probe must be filtered too, and were not. A cross-cohort run is
    method=svd, k=64, fold=0, n_fit=None, max_slides=2000 - it passes every other
    filter, and its erased AUC is high by construction (transfer fails), so it
    dragged every "base" mean upward. Same for the 7-probe-family run, which
    contributed five extra probes to means meant to cover logreg+mlp. Together
    they are why spectral(lam=0) read 0.4312 vs svd 0.5254 when the two subspaces
    are in fact identical to 6 decimal places.
    """
    d = df
    if 'n_fit' in d:
        d = d[d.n_fit.isna()]
    if 'fit_on' in d:
        # fit_on is recorded PER TARGET and equals the target for a normal run,
        # so it is never NaN; cross-cohort is fit_on != target.
        d = d[d.fit_on.isna() | (d.fit_on == d.target)]
    if 'max_slides' in d:
        d = d[d.max_slides == 2000]
    if 'patches' in d:
        d = d[d.patches == 256]
    if 'seed' in d:
        d = d[d.seed == 0]
    if 'probe' in d:
        d = d[d.probe.isin(BASE_PROBES)]
    return d


def hdr(t):
    print("\n" + "=" * 82)
    print(t)
    print("=" * 82)


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out_dir', default='results/sweep_v2')
    ap.add_argument('--probe', default='mlp')
    ap.add_argument('--config_dir', default='configs/sweeps_full',
                    help="only used to report completeness")
    args = ap.parse_args()

    tidy, S, failed = load(args.out_dir)
    if S.empty:
        raise SystemExit(f"no completed results in {args.out_dir}")
    tidy.to_csv(f'{args.out_dir}/_tidy.csv', index=False)
    S.to_csv(f'{args.out_dir}/_summary.csv', index=False)

    P = S[S.probe == args.probe]
    n_cfg = len(glob.glob(os.path.join(args.config_dir, '*.json')))
    print(f"{S.run_id.nunique()}/{n_cfg} configs complete | probe={args.probe} "
          f"| chance=0.50")
    if len(failed):
        print(f"{len(failed)} FAILED:")
        for _, r in failed.iterrows():
            print(f"   {r.run_id}: {str(r.error)[:90]}")

    # -- 1. fold error bars ------------------------------------------------ #
    hdr("1. ERASURE WITH FOLD ERROR BARS  (svd, k=64, virchow2, base settings)")
    d = base_settings(P[(P.method == 'svd') & (P.k == 64) &
                        (P.fm == 'features_virchow2')])
    g = d.groupby('target').agg(
        baseline=('baseline', 'mean'), erased_mean=('erased', 'mean'),
        erased_sd=('erased', 'std'), ci95=('erased', ci95),
        n_folds=('erased', 'size'), min=('erased', 'min'), max=('erased', 'max'))
    print(g.round(4).to_string())

    # -- 2. encoder x fold ------------------------------------------------- #
    hdr("2. ENCODER x FOLD  (is encoder-dependence real, or fold noise?)")
    d = base_settings(P[P.axis.isin(['encoder', 'encoder_x_fold'])])
    if not d.empty:
        piv = d.pivot_table(index='fm', columns='target', values='erased',
                            aggfunc=['mean', 'std'])
        print(piv.round(3).to_string())
        print("\n-- variance decomposition (share of total variance) --")
        for tgt, sub in d.groupby('target'):
            tot = sub.erased.var()
            enc = sub.groupby('fm').erased.mean().var()
            fld = sub.groupby('fold').erased.mean().var()
            print(f"  {tgt:22s} encoder {enc / tot:5.1%}   fold {fld / tot:5.1%}   "
                  f"residual {max(0, 1 - enc / tot - fld / tot):5.1%}")

    # -- 3. significance --------------------------------------------------- #
    hdr("3. SIGNIFICANCE  (paired over folds; vs baseline and vs chance)")
    from scipy import stats
    d = base_settings(P[(P.method == 'svd') & (P.k == 64)])
    for (fm, tgt), sub in d.groupby(['fm', 'target']):
        if len(sub) < 3:
            continue
        t_b = stats.ttest_rel(sub.baseline, sub.erased)
        t_c = stats.ttest_1samp(sub.erased, 0.5)
        print(f"  {fm.replace('features_',''):12s} {tgt:22s} n={len(sub):2d} "
              f"| erased {sub.erased.mean():.3f} "
              f"| vs baseline p={t_b.pvalue:.4f}{'*' if t_b.pvalue < .05 else ' '} "
              f"| vs chance p={t_c.pvalue:.4f}{'*' if t_c.pvalue < .05 else ' '}")

    # -- 4. saturation ----------------------------------------------------- #
    hdr("4. RANK SATURATION  (where k stops helping)")
    d = P[P.axis.isin(['rank', 'rank_x_fold'])]
    if not d.empty:
        piv = d.pivot_table(index='k', columns='target', values='erased', aggfunc='mean')
        print(piv.round(4).to_string())
        print("\n-- best k per target --")
        for tgt in piv.columns:
            print(f"  {tgt:22s} k={piv[tgt].idxmin()}  ({piv[tgt].min():.4f})")

    # -- 5. confound ladder ------------------------------------------------ #
    hdr("5. COLLATERAL BY RELATION  (the confound ladder)")
    t = tidy[(tidy.probe == args.probe) & (tidy.method == 'svd')]
    if not t.empty:
        g = t.groupby('relation').agg(
            control_baseline=('control_baseline', 'mean'),
            control_erased=('control_erased', 'mean'),
            drop_mean=('control_drop', 'mean'), drop_sd=('control_drop', 'std'),
            n=('control_drop', 'size'))
        print(g.round(4).to_string())
        print("\n-- same-slides pairs individually --")
        ss = t[t.relation == 'same-slides'].groupby(['target', 'control']).agg(
            baseline=('control_baseline', 'mean'), erased=('control_erased', 'mean'),
            drop=('control_drop', 'mean'), sd=('control_drop', 'std'),
            n=('control_drop', 'size'))
        print(ss.round(4).to_string())

    # -- 6/7. control validations ------------------------------------------ #
    hdr("6-7. VALIDATION CHECKS")
    lr = P[P.method == 'low_rank']
    if not lr.empty:
        gap = (lr.baseline - lr.erased).mean()
        print(f"  negative control (low_rank) mean drop {gap:+.4f}  "
              f"{'PASS - erases nothing' if abs(gap) < 0.12 else 'FAIL - harness suspect'}")
    sp0 = base_settings(P[(P.method == 'spectral') & (P.spectral_lam == 0.0) &
                          (P.fold == 0)])
    sv = base_settings(P[(P.method == 'svd') & (P.k == 64) &
                         (P.fm == 'features_virchow2') & (P.fold == 0)])
    if not sp0.empty and not sv.empty:
        for tgt in sorted(set(sp0.target) & set(sv.target)):
            a = sp0[sp0.target == tgt].erased.mean()
            b = sv[sv.target == tgt].erased.mean()
            print(f"  spectral(lam=0) vs svd  {tgt:22s} {a:.4f} vs {b:.4f}  "
                  f"{'PASS' if abs(a - b) < 1e-3 else 'DIFFERS'}")

    # -- 8. what predicts erasability -------------------------------------- #
    hdr("8. WHAT PREDICTS ERASABILITY")
    d = base_settings(P[(P.method == 'svd') & (P.k == 64)]).dropna(
        subset=['baseline', 'erased'])
    if len(d) > 5:
        for col in ['baseline', 'dim', 'n_train', 'n_test']:
            if d[col].nunique() > 1:
                r, p = stats.pearsonr(d[col], d.erased)
                print(f"  {col:10s} vs erased AUC   r={r:+.3f}  p={p:.4f}"
                      f"{'  *' if p < 0.05 else ''}")

    # -- 8b. attacker budget ------------------------------------------------ #
    hdr("8b. ATTACKER SAMPLE BUDGET  (how many slides does the attack need?)")
    d = P[P.axis.isin(['attacker_budget', 'budget_x_fold'])]
    if not d.empty:
        g = d.groupby(['n_fit', 'target']).agg(
            erased=('erased', 'mean'), sd=('erased', 'std'),
            eff_k=('effective_k', 'mean'), n_used=('n_fit_used', 'mean'),
            n=('erased', 'size'))
        print(g.round(3).to_string())
        print("\n-- NOTE: k is clamped to n_fit-1, so budget caps attack strength --")
        piv = d.pivot_table(index='n_fit', columns='target', values='erased')
        base_full = P[(P.axis == 'base')].groupby('target').erased.mean()
        print("\n-- erasure vs full-budget baseline --")
        for tgt in piv.columns:
            if tgt in base_full.index:
                print(f"  {tgt:22s} full={base_full[tgt]:.3f}  " +
                      "  ".join(f"n={int(i)}:{v:.3f}" for i, v in piv[tgt].items()))

    # -- 8c. cross-cohort transfer ------------------------------------------- #
    hdr("8c. CROSS-COHORT TRANSFER  (fit on cohort A, attack cohort B)")
    d = P[P.axis == 'cross_cohort']
    if not d.empty:
        g = d.pivot_table(index='fit_on', columns='target', values='erased',
                          aggfunc='mean')
        print(g.round(4).to_string())
        print("\n-- same-cohort reference (fit_on == target) --")
        print(P[P.axis == 'base'].groupby('target').erased.mean().round(4).to_string())

    # -- 9. extremes -------------------------------------------------------- #
    hdr("9. BEST AND WORST CONFIGURATIONS")
    d = P[P.method.isin(['svd', 'spectral'])].copy()
    d['margin'] = d.target_drop - d.collateral.abs()
    print("-- strongest selective erasure --")
    print(d.nlargest(8, 'margin')[
        ['run_id', 'target', 'erased', 'target_drop', 'collateral', 'margin']
    ].round(4).to_string(index=False))
    print("\n-- weakest --")
    print(d.nsmallest(5, 'margin')[
        ['run_id', 'target', 'erased', 'target_drop', 'collateral', 'margin']
    ].round(4).to_string(index=False))

    # -- 10. other axes ------------------------------------------------------ #
    for axis, key in [('method', 'method'), ('spectral', 'spectral_lam'),
                      ('noise_sigma', 'sigma'), ('noise_dropout', 'dropout_p'),
                      ('patches', 'patches'), ('sample_size', 'max_slides'),
                      ('seed', 'seed')]:
        d = P[P.axis == axis]
        if d.empty:
            continue
        hdr(f"10. AXIS {axis}  (rows = {key})")
        piv = d.pivot_table(index=key, columns='target', values='erased', aggfunc='mean')
        print(piv.round(4).to_string())
        cl = d.pivot_table(index=key, columns='target', values='collateral', aggfunc='mean')
        print("\n-- collateral --")
        print(cl.round(4).to_string())

    # -- 9b. R1: random-subspace control ----------------------------------- #
    hdr("9b. RANDOM-SUBSPACE CONTROL  (does targeting the top-k directions matter?)")
    rc = P[P.method.isin(['svd', 'random'])]
    rc = rc[rc.n_fit.isna() & (rc.max_slides == 2000) & (rc.seed == 0)
            & (rc.fit_on.isna() | (rc.fit_on == rc.target))]
    rc = rc[rc.probe.isin(BASE_PROBES)]
    if not rc.empty and 'random' in set(rc.method):
        piv = rc.pivot_table(index='k', columns=['target', 'method'], values='erased',
                             aggfunc='mean')
        print(piv.round(4).to_string())
        print("\nRemoving k RANDOM orthogonal directions deletes the same number of")
        print("dimensions without targeting cohort variance. If the two columns match,")
        print("the top-k selection is doing nothing and the result is about rank alone.")
        for tgt in sorted(set(rc.target)):
            d2 = rc[rc.target == tgt]
            for k in sorted(set(d2.k)):
                a = d2[(d2.k == k) & (d2.method == 'svd')].erased.mean()
                b = d2[(d2.k == k) & (d2.method == 'random')].erased.mean()
                if not (np.isnan(a) or np.isnan(b)):
                    print(f"  {tgt:20s} k={k:4d}  svd {a:.3f}  random {b:.3f}  "
                          f"gap {b - a:+.3f}")
    else:
        print("  (no random-subspace runs yet)")

    # -- 9c. R2: selectivity and destruction at the SAME operating point ---- #
    hdr("9c. SELECTIVITY vs RANK  (do precision and destruction coexist?)")
    sr = P[(P.method == 'svd') & P.n_fit.isna() & (P.max_slides == 2000)
           & (P.seed == 0) & (P.fit_on.isna() | (P.fit_on == P.target))
           & P.probe.isin(BASE_PROBES)]
    if not sr.empty:
        g = sr.groupby('k').agg(target_auc=('erased', 'mean'),
                                baseline=('baseline', 'mean'),
                                collateral=('collateral', 'mean'), n=('erased', 'size'))
        g['target_drop'] = g.baseline - g.target_auc
        g['selectivity'] = g.target_drop / g.collateral.abs().clip(lower=1e-4)
        print(g.round(4).to_string())
        print("\nQuoting collateral from low k and destruction from high k as one result")
        print("overstates the method. This table is the honest operating curve.")

    # -- 9d. R4: is a sub-chance AUC a finding or a small test set? --------- #
    hdr("9d. STATISTICAL POWER  (why some cells fall below chance)")
    pw = P[P.probe.isin(BASE_PROBES)].dropna(subset=['erased', 'n_test'])
    if not pw.empty:
        pw = pw.copy()
        pw['se'] = np.sqrt(0.25 / (pw.n_test / 2))     # AUC SE at true chance
        pw['spans_chance'] = (pw.erased - 0.5).abs() < 1.96 * pw.se
        for tgt, sub in pw.groupby('target'):
            below = (sub.erased < 0.45).sum()
            print(f"  {tgt:20s} n_test {int(sub.n_test.min())}-{int(sub.n_test.max())} "
                  f"(median {int(sub.n_test.median())})   typical SE {sub.se.median():.3f}"
                  f"   cells <0.45: {below}")
        lo = pw.nsmallest(1, 'erased')
        if len(lo):
            r = lo.iloc[0]
            print(f"\n  lowest overall {r.erased:.3f} on {r.target} at n_test={int(r.n_test)}; "
                  f"at true chance the 95% band there is "
                  f"[{0.5 - 1.96 * r.se:.3f}, {0.5 + 1.96 * r.se:.3f}]")
        print(f"  {int(pw.spans_chance.sum())} of {len(pw)} measurements are statistically "
              f"indistinguishable from chance.")
        print("  Sub-chance values are sampling noise on small test sets, not evidence of")
        print("  'better than complete' erasure, and must not be reported as such.")

    # -- 10c. probe ladder ------------------------------------------------ #
    hdr("10c. PROBE LADDER  (does a stronger readout recover the erased signal?)")
    pf = S[S.axis == 'probe_family']   # S, not P: P is fixed to one probe
    if not pf.empty:
        piv = pf.pivot_table(index='probe', columns='target', values='erased',
                             aggfunc='mean')
        bl = pf.pivot_table(index='probe', columns='target', values='baseline',
                            aggfunc='mean')
        order = [q for q in ('logreg', 'linsvm', 'rbfsvm', 'knn', 'rf', 'mlp',
                             'mlp_big') if q in piv.index]
        print("-- erased AUC by probe (chance = 0.50) --")
        print(piv.loc[order].round(4).to_string())
        print("\n-- baseline AUC by probe --")
        print(bl.loc[order].round(4).to_string())
        print("\n-- folds per probe --", int(pf.groupby('probe').fold.nunique().max()))

    # -- fidelity ------------------------------------------------------------ #
    hdr("11. FIDELITY  (centred cosine; raw cosine is meaningless here)")
    d = P.dropna(subset=['centred_cosine'])
    if not d.empty:
        print(d.groupby(['method'])['centred_cosine']
              .agg(['mean', 'std', 'size']).round(4).to_string())

    print(f"\nWrote {args.out_dir}/_tidy.csv and _summary.csv")


if __name__ == "__main__":
    main()
