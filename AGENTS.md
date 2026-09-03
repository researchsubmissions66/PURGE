# AGENTS.md: PURGE

Context for anyone (human or agent) picking up this repo. Written after a full
validation pass. Read the traps section before running anything.

---

## What this project is

A **targeted adversarial attack on representations**: delete one downstream
clinical task from frozen pathology foundation-model embeddings while leaving
unrelated tasks intact. Not organ removal: the goal is task-specific erasure for
a specific organ.

Data is pre-extracted patch features in HDF5 (`/work/hdd/bhwm/...`), d=2560 for
`features_virchow2`. Slide-level MIL probes downstream.

---

## The one rule

**An eraser must be non-invertible.** An invertible map: including the residual
adapter `z' = z + αBAz`: is an information-preserving bijection. It fools the
fixed probe it was trained against; a *retrained* probe recovers the concept
exactly.

This is not theoretical. The original pipeline used exactly that adapter and
reported erasure that did not exist. Measured `σ_min = 2.27e-02` at `rank 2560/2560`, so zero dimensions were annihilated, and the eval AUC confirmed it (BACH→BACH 0.9884).

Guardrails now in place:

* `src/unlearning/audit.py`: `audit_eraser()` reports rank / invertibility.
* `scripts/train_adversarial.py` refuses to launch an invertible eraser without
  `--allow_invertible`.
* `tests/test_erasers.py::test_low_rank_adapter_is_invertible` pins the negative
  result so it cannot silently return.

**Never quote an AUC from a probe that was trained alongside the eraser.** The
co-adapted in-loop probe showed 0.58 while a retrained probe showed 0.99 on the
same eraser. Only from-scratch retrained probes count.

---

## THE RESULT THAT REFRAMES THE PROJECT (2026-09-03)

Cohort-specificity is **not** a blanket property of linear probes. It is graded by
how coarse the concept is. Measured with `scripts/concept_transfer_test.py`,
BACH -> BRACS, both breast, virchow2, fold 0:

| concept | within | transfer | refit | gap | probe cosine |
|---|---|---|---|---|---|
| Normal vs Invasive | 1.000 | **0.9997** | 0.999 | +0.000 | 0.488 |
| Invasive vs InSitu | 1.000 | **0.9839** | 0.997 | +0.016 | 0.404 |
| InSitu vs Benign | 0.996 | 0.8802 | 0.977 | +0.116 | 0.072 |
| Benign vs Normal | 1.000 | 0.8328 | 0.874 | +0.167 | 0.038 |

Perfectly monotone in both the transfer gap and the angle between the two
cohorts' probe directions. Coarse morphology is genuinely encoded and portable.
Fine distinctions are read off cohort-specific geometry: near-orthogonal
directions (cosine 0.04) even though BOTH cohorts support the task on their own
(refit 0.87).

**Do not claim "probe results are cohort artifacts."** That is false for coarse
concepts and the referees will find the counterexample immediately. The claim is:

> A linear probe's within-cohort AUC does not tell you which regime you are in.
> For coarse morphology the direction is portable; for fine distinctions it is
> cohort geometry that happens to correlate with the label. Only a cross-cohort
> transfer test separates the two.

**This subsumes the erasure results rather than competing with them.**
BRACS-atypia (ADH vs FEA) is the finest distinction in the benchmark. It is also
the most completely erased target, the one with the largest fold variance, and
the one whose eraser transfers worst. Same underlying fact, seen three ways.

**And it supplies the mitigation the work previously lacked.** The transfer test
needs no erasure machinery, runs in seconds on cached features, and turns the
finding into something a practitioner can act on before publishing a capability
claim. `scripts/concept_transfer_test.py`.

---

## AUTHORITATIVE RESULTS: full 5-fold sweep (2026-09-02)

`results/sweep_full/`, 276 configs, **every axis crossed with all 5 folds**.
Reproduce with `python scripts/analyze_sweep.py --out_dir results/sweep_full`.
Where a number below disagrees with one further down this file, **this section
wins**: the older ones are fold-0 singletons or predate the analyzer fix in
trap 17.

Base settings: virchow2, affine SVD, k=64, max_slides=2000, mlp probe, chance 0.50.

| target | baseline | erased | sd | n |
|---|---|---|---|---|
| BRACS-atypia | 0.8927 | **0.5792** | 0.091 | 36 |
| PANDA-grade | 0.9105 | **0.7791** | 0.014 | 36 |
| TCGA-LUNG-subtype | 0.9675 | **0.6148** | 0.026 | 31 |

Significant vs baseline on all 8 encoders. Reaches **chance** only for
BRACS-atypia, and only on 6 of 8 (not conch_v15, not virchow2). Encoder explains
27-39% of the variance, fold 14-35%, so encoder-dependence is real but is not
much larger than fold noise for PANDA.

### Rank: the k=64 saturation was an artifact, chance needs k>=256

| k | BRACS | PANDA | TCGA-LUNG |
|---|---|---|---|
| 8 | 0.8200 | 0.8768 | 0.8933 |
| 64 | 0.5834 | 0.7794 | 0.6151 |
| 256 | 0.4953 | 0.5994 | 0.5362 |
| 512 | 0.5280 | **0.5306** | **0.4781** |

All three reach chance at k=512, replicated over 5 folds. Since `k <= n_fit - 1`,
running the attack where it actually works needs **>= 513 fitting slides**.

### Cross-cohort transfer fails, and misfires

| fit_on \\ target | BRACS-atypia | PANDA-grade | TCGA-LUNG |
|---|---|---|---|
| BACH-histology | 0.8983 | 0.9196 | 0.9643 |
| BRACS-atypia | *0.5834* | 0.9068 | 0.9635 |
| PANDA-grade | 0.9014 | *0.7794* | 0.9644 |
| UBC-OCEAN-subtype | 0.8872 | 0.9139 | 0.9496 |

Every one of the sweep's five weakest configurations is a cross-cohort run, each
with NEGATIVE target drop and collateral around +0.09: fitting on the wrong
cohort damages the controls *more than the target*. The subspace is a property of
the cohort's feature geometry, not of the concept.

### Attacker budget (5 folds per row)

| n_fit | BRACS | PANDA | TCGA-LUNG | eff_k |
|---|---|---|---|---|
| 25 | 0.877 | 0.888 | 0.940 | 24 |
| 100 | 0.820 | 0.870 | 0.898 | 64 |
| 250 | 0.723 | 0.851 | 0.818 | 64 |
| 500 | 0.583 | 0.800 | 0.723 | 64 |
| 1000 | 0.583 | 0.772 | 0.615 | 64 |

Nothing happens below ~100 slides. Most of the effect by 500-1000. Beyond that
the cohorts run out (BRACS caps at 445 usable, lung at 834).

### Probe ladder: accessibility, not content (5 folds, 7 families)

| probe | BRACS | PANDA | TCGA-LUNG |
|---|---|---|---|
| logreg | 0.4553 | 0.5648 | 0.5335 |
| mlp | 0.5834 | 0.7794 | 0.6151 |
| mlp_big | 0.6163 | 0.8042 | **0.7594** |

A bigger readout recovers more on every target.

### Other axes

* **Spectral pencil is a strict loss.** lam=0 is best (0.583) and erased AUC rises
  monotonically to 0.856 at lam=4. Pricing controls into the objective only
  weakens the attack. lam=0 == plain SVD to 4 decimals (superset check passes).
* **Methods**: svd 0.583 / svd_plain 0.583 (affine makes no difference to a
  probe) / leace 0.719 / gaussian 0.849 / dropout 0.906 / low_rank 0.897.
* **Seed** changes nothing (0.5834 vs 0.5834). The variance is fold, not seed.
* **Confound ladder**, n=1519: cross-organ +0.0052, same-organ +0.0138,
  same-slides +0.0486.
* **`n_train` correlates +0.798 (p<1e-4) with erased AUC**: bigger cohorts are
  harder to erase. Same fact as the budget curve, seen from the other side.

---

## What works: affine SVD null-space projection

`--unlearn_method svd`, the default. Fit the target cohort's top-k principal
directions, project them out **mean-preservingly**:

```
z' = (z - mu) - ((z - mu) @ U) @ U.T + mu
```

### Organ-level (MLP probe, chance = 0.50, virchow2, fold 0)

| Target | baseline | best SVD | k | mean collateral (5 controls) |
|---|---|---|---|---|
| PANDA (ISUP grading) | 0.7607 | 0.4864 | 256 | −0.005 |
| BACH (histology) | 0.9602 | 0.4524 | 256 | +0.001 |
| BRACS (subtyping) | 0.8431 | 0.5535 | 256* | +0.013 |
| UBC-OCEAN (subtyping) | 0.9579 | 0.4841 | 256 | −0.004 |
| TCGA-LUNG (LUAD/LUSC) | 0.9363 | 0.4661 | 64 | −0.011 |
| TCGA-BRCA (IDC/ILC) | 0.9089 | 0.5400 | 256 | −0.004 |

\* effectively clamped: see traps.

### Task-level, same slides (the unconfounded result)

Target and control share every slide, patient and scanner, so survival cannot be
explained by cohort or organ differences.

| Pair | target | control | verdict |
|---|---|---|---|
| BRACS: erase ADH-vs-FEA, keep malignancy | 0.8841 → **0.4269** | 0.8899 → 0.8061 | works |
| PANDA: erase grading, keep detection | 0.8574 → **0.4832** | 0.9168 → 0.7861 | bleeds 13 pts |

**Selectivity depends on task entanglement.** Grading presupposes detection, so
removing one damages the other. Distinct tasks (atypia subtyping vs malignancy)
separate cleanly. This is the most interesting scientific finding in the repo.

---

## Sanity checks: all passed

`scripts/sanity_check_erasure.py`. Each is built to fail if the result is an
artifact.

| Check | Result |
|---|---|
| Random subspace, same rank | 0.9404 vs baseline 0.9363: **no effect** |
| Wrong cohort's subspace | 0.9341 vs 0.9363: **no effect** |
| Permutation null | erased AUC inside the chance CI on all 3 targets |
| High-capacity probe (1024-512) | 0.6218 / 0.5483 / 0.2569 vs baselines 0.93 / 0.98 / 0.92 |
| Class separation along erased dirs | 2.96 → 3.7e-06 |
| Patient leakage | 0 overlap |

Only the *target's own* subspace works. That is the core specificity evidence.

**Caveat to quote alongside:** erasure is strong, not always total. TCGA-LUNG
recovers to 0.6218 under a high-capacity probe. Report that next to 0.4661, not
instead of it.

---

## What does NOT work (do not re-litigate without new ideas)

**The code for everything in this section has been DELETED.** The findings and
numbers are kept here so the experiments are not repeated; the implementations are
gone so they cannot be picked up by accident. Removed: `relevance.py`,
`iterative.py`, `reconstruct.py`, `svd_plus.py`, `projection.py`,
`train_adversarial.py`, `launch_matrix.sh`, and 26 tests.

Still present, deliberately: `low_rank.py` (the invertible negative control, whose
test pins the original bug) and `spectral.py` (lambda=0 IS affine SVD).

### 1. Adversarial learned eraser: abandoned

`scripts/train_adversarial.py` with `ProjectionEraser`. The min-max does not
converge; in-loop probe accuracy climbs to 0.988 while the adversarial loss rises.

Initializing from the SVD subspace does not save it: **it starts on the answer
and walks away**:

| Target | svd_k64 drop | projection_svdinit_r64 drop |
|---|---|---|
| PANDA | +0.2399 | −0.0190 |
| BACH | +0.3605 | −0.0184 |
| UBC-OCEAN | +0.3530 | +0.0523 |

Code is kept and working; the approach is not recommended.

### 2. Relevance-weighted direction selection, a trade-off knob, not an upgrade

`src/unlearning/relevance.py`. Ranks directions by target relevance per unit of
(control damage + variance removed). 2×2 ablation, PANDA grade/detect, MLP:

| config | target ↓ | detection ↑ |
|---|---|---|
| plain SVD | **0.4832** | 0.7861 |
| relevance λc=0 λd=0 | 0.6646 | 0.7532 |
| relevance λc=0 **λd=1** | 0.8579 | 0.9050 |
| relevance **λc=1** λd=0 | 0.6465 | **0.8474** |

* `λ_c` **works**: +9.4 points of control preservation, mechanism confirmed.
* `λ_d` is **fatal**: erasure goes to zero. The target signal lives in the
  high-variance directions, so penalising them removes the attack. **Default is
  now 0.**
* Neither rescues the core deficit: ranking directions *individually* erases less
  than taking the cohort's principal span. Fisher scores each direction in
  isolation, so signal spread across individually-mediocre directions is invisible.

Swapping Fisher for HSIC (`--relevance_criterion hsic`) does **not** help
(0.7519 vs Fisher's 0.7276). The criterion's linearity was not the cause.

Keep it as an ablation row: it answers "why not pick directions more cleverly?"

### 3. Reconstruction head (distortion masking): failed, but informative

`src/unlearning/reconstruct.py`. `z' = mu + h + U g(h)` with `g` predicting the
erased coefficients from the preserved part `h` alone. Provably cannot leak (DPI),
but measured:

| goal | result |
|---|---|
| improve cos(z, z') | 0.8606 -> 0.8758 (negligible) |
| hide the spectrum signature | near-zero eigenvalues 2448 -> 2448 (no change) |
| preserve erasure | 0.4661 -> **0.7309** (much worse) |

Two lessons:

* **The stealth premise was wrong at this sample size.** With n=470 and d=2560 the
  covariance is already rank-deficient by ~2090, so removing k=64 leaves no visible
  spectrum signature to mask.
* **It accidentally became a better attack evaluator.** `g(h)` adds no information
  (DPI), so the probe going 0.4661 -> 0.7309 means `h` held ~0.73-AUC of target
  signal all along and the standard probe simply could not extract it.

Always fit the head on train and evaluate held-out: in-sample it reaches 100% of
erased-coefficient variance and effectively undoes the projection.

### 4. Iterative INLP: failed

`src/unlearning/iterative.py`. Remove the directions a probe uses, refit, repeat.
At equal rank 64: TCGA-LUNG 0.7837 vs SVD's 0.4661; PANDA 0.8181 vs 0.4832.

### 5. SVD refinements: one win, two neutral/negative

`src/unlearning/svd_plus.py`.

| refinement | axis | verdict |
|---|---|---|
| affine (mean-preserving) | how removal is applied | **WIN: now the default** |
| spectral pencil (lambda) | control-aware variance | best collateral knob; costs erasure |
| bootstrap stabilisation | estimation noise | neutral (0.4661 -> 0.5056 mlp) |
| automatic k (parallel analysis) | the k knob | **worse**: picks k=19-35, erasure drops to 0.6341 |
| background whitening | the metric | **failed**: no erasure at all (0.9309) |

* **Whitening fails for an n<<d reason**: the background covariance is rank
  deficient, so its inverse blows up along the null space and the top generalised
  eigenvectors are numerical artifacts. Diagnostic tell: mean variance ratio 2547.8.
* **Auto-k**: parallel analysis correctly finds ~19 directions with above-chance
  structure, but erasure needs to remove more than the statistically significant
  PCs. An earlier apparent win came from a BUGGY criterion (bootstrap stability)
  that happened to pick k~91; the real content was only "k~100 beats k=64 on
  PANDA", a tuning observation.
* **Do not use bootstrap stability to choose k.** It measures whether an estimate
  is reproducible, not whether a direction carries structure; when n is large
  relative to d, pure noise directions are highly stable (returned k=11 for 5
  planted directions).

### 6. Spectral pencil: the cleanest knob, still not a free lunch

`src/unlearning/spectral.py`. Writes affine SVD as the trace objective it already
solves, then prices the controls into it:

    L_lambda(U) = tr( U^T (Sigma_t - lambda * mean_c Sigma_c) U )

Optimum = top-k eigenvectors of the pencil. **lambda = 0 reproduces plain SVD
exactly** (verified: 0.4832/0.4832 and 0.4661/0.4661), so it is a provable superset.

PANDA-grade, MLP probe:

| method | target | detection (same slides) | SDS |
|---|---|---|---|
| plain SVD (lambda=0) | **0.4832** | 0.7861 | 7.33 |
| spectral lambda=0.5 | 0.6165 | **0.8555** | **10.41** |
| relevance lambda_c=1 | 0.6465 | 0.8474 | 7.09 |

* **Best available collateral knob**: beats the relevance selector on BOTH axes at
  once - the only method that has managed that.
* Still trades erasure for preservation; plain SVD (lambda=0) remains best for raw
  erasure. On TCGA-LUNG, where collateral was already ~0, lambda>0 is strictly worse.
* **Use the difference, never the ratio.** The Rayleigh form
  u^T Sigma_t u / u^T Sigma_c u is background whitening, which failed completely at
  n<<d (singular inverse, mean ratio 2547.8). The pencil needs no inverse.
* No negative eigenvalues at any lambda tested (0/64), so the objective never
  degenerated into "avoid control variance at any cost".
* `spectral_erasure_loss()` is the differentiable form - the route to plan section 19
  (poison the encoder itself rather than bolting on a projection). Untested.

### 7. Quadratic / optimal-transport erasure: fails structurally

`QuadraticEraser` (vendored, previously never run). Equalises class-conditional
means AND covariances by transporting each class to the Wasserstein barycenter -
exactly the second-order structure LEACE leaves behind. It does not work here.

Diagnostic (TCGA-LUNG, fit on 487 train, applied to 113 test):

| m | cond(Sigma_c) | TRAIN class-mean spread | TEST class-mean spread |
|---|---|---|---|
| 16 | 4.3e+01 | 2.868 -> **0.000** | 2.766 -> 1.327 (-52%) |
| 32 | 2.4e+02 | 2.921 -> **0.000** | 2.926 -> 1.471 (-50%) |
| 64 | 8.6e+02 | 2.931 -> **0.000** | 2.958 -> 1.530 (-48%) |

**Exact on the fitting data, ~50% on held-out data, flat across a 20x range in
condition number and identical with `shrinkage=False`.** So it is NOT ill-
conditioning, NOT shrinkage, and NOT the fitting sample size.

The transform is `(x - mu_c^train) A_c + mu_global` with `A_c` built from
`Sigma_c^{-1/2}`. That inverse amplifies the LOW-VARIANCE directions - precisely
where the *test* class-mean estimate is noisiest (n_test = 113, ~56/class). The map
removes the train-estimated mean exactly and inflates whatever the test mean
disagrees by.

**This is a property of transport-based erasure, not of this implementation.** Any
method that whitens by an estimated class covariance inherits it - including
kernelised LEACE / random-Fourier-feature erasure, which has the same inverse in
the RKHS. Do not build those expecting a different outcome.

Erasure quality, quadratic-in-principal-subspace vs plain affine SVD:

| task | quadratic (best) | plain affine SVD |
|---|---|---|
| TCGA-LUNG | 0.5938 (m=128) | **0.4661** (k=64) |
| PANDA-grade | 0.4739 (m=64), 0.4786 (m=32) | 0.4832 (k=64) |

The AUC tracks `m` (how many dimensions are restricted) while the mean-spread
reduction stays flat at ~48% - i.e. **the erasure comes from the subspace
restriction, not from the transport**. Quadratic at m=128 (0.5938) is
indistinguishable from plain SVD at k=256 (0.5951).

One unexplored positive: on PANDA, quadratic reaches chance at m=32 where plain SVD
needs k=64. If that survives a collateral measurement it would mean equal erasure
for half the destroyed dimensions. Untested, and oracle-labelled either way.

**Oracle constraint.** `QuadraticEraser.__call__(x, z)` needs the label per sample
and maps each class to a shared barycenter, so it is NOT a fixed transform and is
undefined off-cohort. It cannot be a released artefact; it only fits a
dataset-poisoning framing where the attacker labels what they publish.

### 8. LEACE: exact linearly, leaks nonlinearly

Hits **exactly 0.5000** on a logistic probe every time (its guarantee, and it
holds once the bias is applied correctly). Against an MLP: 0.6823 / **0.9638** /
0.7299 / 0.8431: for BACH it erases nothing. Correct statement is "linear
erasure does exactly what it promises and nothing more," not "linear erasure
fails."

---

### 9. Nonlinear bottleneck + HSIC: fails, and generalises the mechanism

`src/unlearning/bottleneck.py`. `z' = dec(enc(z))`, m < d, trained on fidelity
(target + all controls) plus `lambda_t * HSIC(enc(z), y_target)`. Chosen because it
avoids BOTH known mechanisms: it selects no directions and inverts no covariance.
It fails anyway.

| | TCGA-LUNG | PANDA-grade |
|---|---|---|
| baseline | 0.9335 | 0.8496 |
| **svd_k64** | **0.5505** | **0.5121** |
| bottleneck m=64 | 0.8211 | 0.8672 |
| bottleneck m=256 | 0.8754 | 0.8744 |
| bottleneck m=512 | 0.8358 | 0.8775 |

On PANDA the drop is NEGATIVE. HSIC did optimise (0.015 -> 0.001, -93%); it simply
is not a strong enough counterweight.

**MECHANISM 3: reconstruction preserves what erasure must destroy.** Final fidelity
0.019 - the autoencoder is 98% lossless, and a lossless map erases nothing by
definition. Confirmed on a purpose-built synthetic case where the bottleneck should
excel (class sets the RADIUS in a 2-D plane, so both classes share a mean and every
principal direction, and no projection can reach it): 0.973 -> **0.998**. It removes
nothing even there. Pinned by
`tests/test_erasers.py::test_reconstruction_objective_defeats_erasure`.

This mechanism retroactively explains two earlier failures catalogued separately:
the reconstruction head (#3) and the spectral encoder objective, which relocated
signal rather than destroying it. **Any objective containing a fidelity term is
simultaneously asking to keep and to remove the same information.**

Incidental findings worth keeping:
* A 64-d nonlinear code reconstructs 2560-d Virchow2 embeddings to 2% error -
  these representations have very low intrinsic dimension.
* Centred cos(z, z') 0.91-0.96, vs ~0.39 for affine SVD. Far more faithful, and
  entirely on the wrong end of the trade-off.

## The central negative result

**Eleven attempts have failed, under THREE mechanisms.**

| mechanism | kills |
|---|---|
| task info is distributed, not concentrated | 6 selection methods |
| `Sigma^-1/2` does not generalise out of sample | transport family, incl. kernelised LEACE |
| reconstruction preserves what erasure must destroy | reconstruction head, spectral encoder objective, bottleneck |

Affine SVD is not a baseline that resisted improvement - it is the only
construction that RESOLVES the third tension, by deleting a specific subspace
outright instead of trying to preserve and remove at once. Its crudeness is why it
works.

**Six independent attempts at smarter direction selection all failed to beat
plain variance ranking on erasure strength:**

| approach | selects by | vs plain SVD |
|---|---|---|
| adversarial ProjectionEraser | gradient descent on the subspace | worse |
| relevance, Fisher criterion | per-direction linear discriminability | worse |
| relevance, HSIC criterion | per-direction nonlinear dependence | worse |
| iterative INLP | directions a probe actually uses, iteratively | worse |
| background whitening | variance ratio vs a background cohort | worse (singular inverse) |
| spectral pencil, lambda>0 | variance minus control variance | worse erasure, better collateral |

**Interpretation: the target task's information is distributed across the target
cohort's principal subspace, not concentrated in a few discriminative directions.**
Bulk removal of the principal span works; surgically chosen directions do not. Any
new method premised on "find the directions that carry the label" should expect to
lose too - propose one only with a reason it escapes this.

This is NOT simply "SVD removes cohort identity". If it were, both PANDA tasks
would die together; instead grading drops to 0.48 while detection holds at 0.79 on
the SAME slides. The top principal directions appear to carry fine morphological
variation (which grading needs) more than the coarse tumour-presence signal (which
detection needs).

## The ceiling: k is capped by n, and the curve saturates long before it

Extended sweep (MLP probe, affine SVD, fold 0):

| k | TCGA-LUNG | PANDA-grade |
|---|---|---|
| 16 | 0.7542 | 0.7158 |
| 64 | **0.4661** | 0.4832 |
| 128 | 0.4834 | **0.4300** |
| 256 | 0.5951 | 0.4364 |
| 512 | 0.4873 | 0.4685 |
| 1024 | 0.4873 | 0.4685 |

**k=512 and k=1024 are IDENTICAL** because `svd_subspace` clamps to n-1: TCGA-LUNG
has 487 train slides (cap 486), PANDA 349 (cap 348). **You cannot remove more
directions than you have samples.** With d=2560 and n~500 that is a permanent
ceiling of ~19% of the space for WSI cohorts - a structural property of the regime,
not a hyperparameter.

**The curve saturates far below that ceiling.** Erasure drops fast to k~64-128 then
oscillates in 0.43-0.60 and never improves. Removing 7x more directions buys
nothing. Collateral does NOT grow either (TCGA-LUNG controls hold 0.90-0.92 at
maximum k; PANDA-detect sits at 0.75-0.80 throughout, flat in k). **There is no
erasure/collateral trade-off curve past k~64 - the method simply saturates.**

Composition `svd_k -> LEACE` is within noise of SVD alone (best case PANDA
svd256_then_leace 0.3685, but that is BELOW chance on ~100 test samples, i.e.
inversion noise). LEACE has nothing left to remove after the projection.

**Conclusion: the residual is irreducible within this family.** Three independent
lines agree - more directions do not help, a high-capacity probe recovers 0.62, and
nonlinear features of the surviving subspace recover 0.73. The surviving signal is
NOT in the principal span, so principal-span removal cannot reach it.

## CORRECTION (2026-09-02): the MIL pipeline was attacking with a WEAK eraser

Measured on TCGA-LUNG, real fold-0 patient split, logreg on slide means,
clean = 0.9540:

| eraser | erased AUC | drop |
|---|---|---|
| patch-fit, 300 slides: **the one behind the ABMIL claim below** | 0.7975 | 0.157 |
| patch-fit, 1000 slides (refit) | 0.9172 | 0.037 |
| **slide-mean fit on the fold's train split (pooled protocol)** | **0.5355** | **0.419** |

Subspace overlap between a patch-fit U and a slide-mean-fit U is only 0.698 mean
principal cosine. They are different subspaces, not noisy versions of one.

Two consequences:

1. **The ABMIL comparison below was run against roughly half the attack.** Both
   models saw the identical transform, so the RATIO is still internally valid,
   but the absolute drops are not comparable to anything in `results/sweep_full`,
   and presenting them side by side overstates how much of the real attack ABMIL
   survives. It must be re-measured with a slide-mean eraser.

2. **Fitting the patch eraser BETTER makes it WEAKER**: 1000 slides (0.9172) is
   worse than 300 (0.7975). A larger patch sample estimates the patch covariance
   more faithfully, and patch covariance is dominated by within-slide texture,
   stain and position, which is not where slide-level label signal lives. Same
   direction as the sweep's `n_train` vs erased-AUC correlation of +0.798.
   Patch-fitting is not an under-tuned attack, it is the wrong protocol.

The projection itself is fine: residual sd inside the removed subspace is 6e-6
across slides, i.e. collapsed to the constant `mu @ U` exactly as intended. The
defect is WHICH subspace, not whether it is removed.

Fix: fit the MIL eraser on the target task's training-split SLIDE MEANS, per
fold, and apply it at patch level. Erasure and mean pooling are both linear and
commute, so MeanMIL then reproduces the pooled number exactly, which gives the
MIL sweep a checkable anchor it never had, and ABMIL/TransMIL measure genuine
architectural resistance to the same attack the rest of the paper reports.

Patch-fitted MIL results are quarantined in `results/_patchfit_mil/`.

---

## THE POOLED PROXY IS A MEANMIL RESULT - ABMIL RESISTS 3x

Identical eraser (LUNG k=64), identical slides, TCGA-LUNG, real patch bags,
BOTH FOLDS COMPLETE:

| fold | MeanMIL drop | ABMIL drop |
|---|---|---|
| 0 | 0.9726 -> 0.7684 (+0.204) | 0.9825 -> 0.9164 (+0.066) |
| 1 | 0.9712 -> 0.7448 (+0.227) | 0.9866 -> 0.9425 (+0.044) |
| **mean** | **+0.2154** | **+0.0551** |

**ABMIL resists 3.9x.** Erased ABMIL sits at 0.92-0.94 against baselines of
0.98-0.99: roughly five points. Against attention-based MIL this is not an attack.

**Attention pooling routes around the erasure.** ABMIL re-learns which patches to
attend to and finds ones where the signal survives; mean pooling cannot.

**Consequence: every pooled-feature result in results/sweep_v2 is a MEANMIL
result.** The proxy's guarantee is that erasure commutes with mean pooling - which
is exact, and which is precisely why it does NOT extend to ABMIL. Reporting pooled
numbers as ABMIL numbers would overstate the attack roughly threefold, and a
reviewer running ABMIL would find it immediately.

**Why attention defeats it.** The eraser removes a fixed subspace from every
patch. Mean pooling then averages patches that have all lost the same directions,
so the slide vector is genuinely degraded. ABMIL RE-LEARNS ITS ATTENTION after
erasure and concentrates on whichever patches still carry signal - the information
was never removed from the bag, only from a subspace of each patch.

This is the same phenomenon as the residual chased all session (0.47 standard probe
/ 0.62 high-capacity / 0.73 given nonlinear features): **subspace removal does not
destroy information, it makes it harder to reach, and a model with a learned
pooling mechanism reaches it.**

Caveats: this eraser is the weaker patch-fitted one
(300 slides), so absolute values may move - but both models saw the IDENTICAL
transform, so the ratio stands. ABMIL's higher baseline (0.9825) gives it more
headroom, though not enough to explain 3x.

**Any headline claim must be measured with ABMIL, not the proxy.** Use the proxy
for breadth (it is ~100x cheaper) and confirm every claim with MIL.

### NOTE: the two pipelines fit erasers differently

`quick_validate`/`run_config` fit on the target TASK's train split using SLIDE-MEAN
vectors. `fit_unlearner.py` fits on an ORGAN using PATCH-level features. They are
not interchangeable, and cross-pipeline magnitudes are not comparable:

| | pooled | MIL |
|---|---|---|
| fitted on | task train split | organ |
| features | slide means | patches (64/slide) |
| slides | ~841 | 300 |
| TCGA-LUNG result | 0.9619 -> 0.6070 | 0.9726 -> 0.7684 |

Only WITHIN-pipeline comparisons are valid until this is unified.

## THE THREAT MODEL HAS TWO BUDGETS

When is the attack applied? **Both train and test** - the eraser is fitted on the
attacker's data, then applied to everything the victim sees. That is POISONING
(poisoned dataset, or poisoned encoder), not evasion. Fitting on train and
evaluating on held-out test keeps the attacker away from the victim's eval data.

### Attacker budget (`n_fit`) - the attack is not cheap

virchow2, k=64 requested, max_slides=2000:

| n_fit | effective k | TCGA-LUNG | PANDA-grade | BRACS-atypia |
|---|---|---|---|---|
| (baseline) | - | 0.9619 | 0.8978 | 0.8789 |
| 25 | **24** (clamped) | 0.9214 | 0.8922 | 0.8070 |
| 250 | 64 | 0.8479 | 0.8468 | 0.6778 |
| full (373-1148) | 64 | 0.6070 | 0.7685 | 0.4313 |

**Budget caps attack strength through the rank clamp**: `k <= n_fit - 1`, so an
attacker with 25 slides can remove at most 24 directions however large a k they
request. Budget and rank are NOT independent knobs. `run_config.py` records
`effective_k` for this reason - never trust the requested k.

At 25 slides the attack barely functions. It needs hundreds to low thousands of
slides - and for the supervised variants, LABELLED ones. State that.

### Victim budget - cuts the other way, and we had been measuring the weak end

Identical eraser and k, only the victim's training data changes:

| target | erased @600 slides | erased @2000 slides |
|---|---|---|
| TCGA-LUNG | 0.4661 | **0.6070** |
| PANDA-grade | 0.4832 | **0.7685** |
| BRACS-atypia | 0.4269 | 0.4313 |

**More victim data -> weaker erasure.** A fixed 64-direction removal does not keep
up with a probe trained on 1148 slides instead of 349. Every headline number
before this was measured at the victim's WEAKEST setting - the same class of
optimism as reporting fold 0.

**So the attack's success is the GAP between the two budgets.** A well-resourced
victim substantially defeats it. A reviewer training on all 10,616 PANDA slides
would find this immediately.

### Cross-cohort transfer: MEASURED, AND IT FAILS COMPLETELY

`fit_on` fits the eraser on one cohort and attacks another (virchow2, k=64).
Target-AUC drop:

| eraser fitted on | -> TCGA-LUNG | -> PANDA-grade | -> BRACS-atypia |
|---|---|---|---|
| **same cohort** | **+0.470** | **+0.374** | **+0.457** |
| BACH-histology | +0.000 | -0.023 | +0.084 |
| BRACS-atypia | -0.000 | -0.022 | (same) |
| PANDA-grade | +0.003 | (same) | -0.002 |
| UBC-OCEAN-subtype | +0.021 | -0.023 | +0.037 |

**Every off-diagonal cell is within noise of zero; several are negative.** BACH ->
BRACS is the same ORGAN and still transfers nothing (+0.084).

**Consequences.**
* The **poisoned-encoder** story (plan section 23) is DEAD. It requires a released
  E_theta' to erase the target on data the attacker never saw. It does not.
* The defensible story is the **poisoned dataset**: the attacker must control the
  specific cohort the victim trains on.
* This also explains the encoder-dependence seen earlier - the eraser fits
  COHORT-SPECIFIC structure, not a transferable concept direction. Consistent with
  the central finding that task information lives in *that cohort's* principal span.

**The honest threat model:** an adversary who controls a specific labelled cohort
(hundreds to thousands of slides) can degrade one downstream task ON THAT COHORT
while leaving other tasks on the same slides intact. It does not transfer to other
cohorts, and it weakens as the victim's training set grows.

plan.md section 23 needs revising before anything is written.

## How strong is the erasure, honestly

Report all three numbers. They are the same eraser, different probes:

| probe | TCGA-LUNG |
|---|---|
| standard MLP | 0.4661 |
| high-capacity MLP (1024-512) | 0.6218 |
| MLP given nonlinear features of the surviving subspace | 0.7309 |

**The attack makes the target task hard to learn, not impossible.** For a security
claim that distinction matters; a determined adversary reaches 0.73. Do not write
"erased to chance" without qualification.

## Encoder-side poisoning (plan section 19)

`scripts/poison_encoder.py` + `src/unlearning/lora.py`. Fine-tunes LoRA adapters in
the last N blocks so the ENCODER emits erased embeddings, with no external
transform - the supply-chain threat model, where the released artefact looks
ordinary.

    min_phi  tr(U^T Sigma_t(phi) U)  +  mu * ||Z(phi) - Z_0||^2 / ||Z_0||^2

Alternating, NOT adversarial: U is re-solved in closed form every `--refit_every`
steps, then phi descends. Both steps decrease the same objective. Every min-max
variant in this repo diverged, so do not reintroduce one here.

Scale: 8 wrapped layers, **98,304 trainable of 87.5M params (0.112%)** on PLIP.

### Measured: HSIC works, spectral does not (PLIP, synthetic patches)

| objective | probe AUC | cos(z0, z1) |
|---|---|---|
| spectral (geometry) | 0.9503 -> 0.9339 | 0.8726 |
| **hsic (dependence)** | **0.9514 -> 0.8200** | **0.9954** |

**Why spectral fails, and it is a general lesson.** A projection erases because it
is RANK DEFICIENT. An encoder is a FULL-RANK map: told to avoid a 32-dim subspace
it relocates the signal into the other 736 and satisfies the loss without losing
information. Measured directly - it evacuated span(U) (variance 0.6105 -> 0.2406)
while the probe lost 1.6 points. Optimising geometry is not optimising information.
External projection and internal poisoning are therefore NOT interchangeable.

**Erasure replicates. The stealth claim does NOT - it was a metric artifact.**

Five seeds at r=32 / 8 blocks / mu=0.5 (each seed varies both data and init):

| seed | baseline | after | cos raw | **cos centred** | shared-mean energy |
|---|---|---|---|---|---|
| 0 | 0.9514 | 0.4547 | 0.9940 | n/a | n/a |
| 1 | 0.9533 | 0.4790 | 0.9942 | **0.0220** | 0.9885 |
| 2 | 0.9637 | 0.5916 | 0.9937 | **0.0327** | 0.9877 |
| 3 | 0.9647 | 0.5175 | 0.9936 | **0.0278** | 0.9874 |
| 4 | 0.9293 | 0.4914 | 0.9937 | **0.0284** | 0.9875 |

erasure mean **0.5068, sd 0.0525** - genuine, replicated, at chance.

**98.8% of PLIP's embedding norm is a shared mean vector.** Raw cosine 0.9940 was
measuring that constant. Centred, cosine to the original is **0.028**: the
informative part of the embedding is essentially ORTHOGONAL to where it started.

| method | erasure | centred fidelity |
|---|---|---|
| external affine SVD | 0.47 | ~0.39 (sqrt of 0.154 retained variance) |
| internal HSIC LoRA | 0.507 +- 0.053 | **0.028** |

**The external projection is ~14x more faithful.** An earlier entry here claimed
internal poisoning "dominates" - that was WRONG and is retracted. Internal
poisoning erases by demolishing the representation, which is the opposite of the
supply-chain property plan section 23 needs.

Capacity is also NOT monotonic (0.8363 / 0.4405 / 0.6077 / 0.5663 at 0.112% /
0.891% / 1.331% / 2.627% trainable), and mu does not control fidelity - centred
cosine stayed ~0.028 across a 10x change in mu, so the penalty is not binding.

### DATA BLOCKER - read before planning work here

**The original histology images are NOT on this filesystem.** Only extracted HDF5
features and contour thumbnails. `_config_coords.json` records `wsi_source` as a
path on a DIFFERENT machine. Encoder poisoning cannot be evaluated on PANDA/BACH/
BRACS/TCGA here, because backprop needs pixels.

What exists locally:

| asset | status |
|---|---|
| raw WSIs (`/work/hdd/bhwm/Datasets/TCGA-*`, `UBC-OCEAN_slides`) | present |
| patch coordinates | present |
| **openslide / cucim** (to cut patches from WSIs) | **MISSING** |
| **peft** | MISSING (hence the hand-rolled LoRA) |
| PLIP (`/work/hdd/bhwm/dchanda/model_cache/plip`) | present, fine-tunable |
| Virchow2 / UNI / GigaPath weights | not cached; gated, need approved HF access |

To run for real: supply `--image_root <root>/<label>/<file>`. `--smoke_test` uses
synthetic patches and validates only that gradients flow through a real ViT and the
loss moves the embedding spectrum - it says NOTHING about pathology.

**Do not trust a smoke test whose baseline AUC is 1.0.** The first synthetic
generator used a deterministic per-class template and PLIP separated it perfectly;
erasure from a starting point of 1.0 is meaningless. `--difficulty` now scales the
class signal against per-image noise; aim for a baseline of 0.80-0.95.

## Traps that have already cost time

1. **`svd_subspace` silently clamped `k`.** BRACS-atypia has ~133 training slides,
   so "k=256" was really k=132. Both selectors now emit a `RuntimeWarning`. Check
   for it in logs before trusting a rank column.

2. **SDS rewards timidity.** With collateral ≈ 0 the `eps=1e-3` denominator makes
   SDS explode. A relevance eraser scored **46.5 vs SVD's 38.8 while erasing far
   less** (0.726 vs 0.466). Never rank methods by SDS without the target drop.

3. **The affine form maps the erased subspace to a CONSTANT, not zero**
   (`z' @ U == mu @ U`). A check of `||z' @ U|| == 0` will fail. Test that the
   component along `U` has zero *variance*.

4. **Audit must probe in float64.** An affine eraser computes `(z−μ)…+μ`; in
   float32 that cancellation costs ~1e-4 precision and swamps the rank threshold.
   Do **not** fix this by loosening the threshold, an invertible low-rank adapter
   has condition number ~3e4 and a loose cutoff would call it rank-deficient,
   defeating the audit.

5. **Two feature-cache formats exist.** Older `.npz` store `y` + `classes`; newer
   store raw `labels`. Loaders handle both; new code must too.

6. **Chance is 0.50 for macro one-vs-rest AUC**, regardless of class count. Not
   `1/num_classes`.

7. **Undefined AUC must never return 0.5.** A placeholder is indistinguishable
   from a genuine chance result: the exact quantity this project measures.
   `macro_ovr_auc` returns `None` plus a reason.

8. **`torch.linalg.lstsq` defaults to the `gels` driver, which assumes full
   rank.** After a projection the input is rank-deficient and it silently returns
   garbage (observed: R^2 ~ 0 where the truth was ~0.9, which made a test pass for
   the wrong reason). Use `torch.linalg.pinv`, as `tests/test_erasers.py::linear_r2`
   does.

9. **Do not plant one label in two coordinates to build a fixture.** They become
   collinear, so removing either removes both, and the "missed signal" you meant to
   create does not exist. This broke two separate tests.

10. **New nn.Parameters do not inherit the model's device.** LoRA adapters were
    created with the default factory and landed on CPU while the model was on CUDA
    (`mat2 is on cpu`). Build them on `base.weight.device`/`dtype`.

11. **NEVER report raw cosine on foundation-model embeddings.** They are
    dominated by an enormous shared mean - measured 98.8% of ||Z||^2 for PLIP. Raw
    cosine measures that constant, not fidelity. This misled this project TWICE, in
    opposite directions: plain SVD looked catastrophic at cos 0.43 (the mean was the
    problem), and encoder poisoning looked near-perfect at cos 0.994 (centred: 0.028).
    Always centre before computing similarity, and report the shared-mean energy
    fraction alongside.

12. **Same-slide task pairs go degenerate under oracle erasure.** PANDA-grade
    excludes ISUP_0, so on the grade-labelled subset every slide is cancer and
    PANDA-detect is constant. Same for BRACS atypia vs malignancy. Oracle methods
    cannot be evaluated for collateral on these pairs.

13. **`run_id` does not identify a configuration.** It is built from the VARIED
    keys only, so editing a base setting in `configs/sweep.yaml` (max_slides
    600 -> 2000, n_splits, the control list) leaves the filename unchanged while
    the experiment changes. Results are then silently reused and the analysis
    pools incomparable settings - 17 files were contaminated this way, covering
    `base`, every `k-*` config and the method arms. `run_config.py` now compares
    an existing result's config against the current one on 16 settings keys and
    recomputes on mismatch. Quarantined copies are in `results/_stale_sweep_v2/`.

14. **`/tmp` is node-local on Delta.** A file written on the login node is not
    visible to a compute job; a batch list placed there reads as empty and the job
    reports "0 items processed" with no error. Keep anything a job must read on the
    shared filesystem.

15. **Feature coverage is not uniform across encoders.** `conch_v15` and `virchow`
    have ZERO BACH slides while every other encoder has 400; `uni_v2` has 5848
    TCGA slides against everyone else's ~2169. Check FILE COUNTS, not directory
    existence - an empty directory passes an `isdir` test and then fails at load.

16. **BRACS-atypia is underpowered**: 167 cached slides of 4,539 available, ~133
   training. Every method is unstable there (0.90 → 0.20 → 0.47 across k). Prefer
   the PANDA pair (~470 training slides) for same-slides claims.

17. **A filter helper that misses one axis silently corrupts every number that
   uses it.** `analyze_sweep.base_settings()` filtered `n_fit`, `max_slides`,
   `patches` and `seed` but not `fit_on` or `probe`. A cross-cohort run is
   `method=svd, k=64, fold=0, n_fit=None, max_slides=2000`, it passes every
   remaining filter, and its erased AUC is high *by construction* (transfer
   fails), so all four of them were being averaged into the base means. The
   7-probe-family run leaked five extra probes the same way. Effect: headline
   erased AUCs read 0.5778 / 0.7932 / 0.6660 when the correct values are
   **0.5451 / 0.7772 / 0.6156**: the attack was reported as *weaker* than it is, and the `spectral(lam=0) == svd` check read 0.4312 vs 0.5254 and was
   recorded for a day as an unexplained method failure. It was not: the two
   subspaces agree to `min singular value 1.000000` on real features at k=16,
   64 and 256, and once the filter is fixed the check passes on all three
   targets. **A validation check that fails is at least as likely to indict the
   analyzer as the method.** Verify the mechanism directly before believing an
   aggregate.

18. **`fit_on` is recorded per target and equals the target for a normal run.**
   It is therefore never NaN, and `d[d.fit_on.isna()]` drops every row rather
   than keeping the non-cross-cohort ones. The correct test is
   `fit_on.isna() | (fit_on == target)`.

19. **Never edit a shell script while bash is executing it.** Bash re-reads the
   file from its current byte offset, so inserting lines shifts everything after
   the cursor and execution resumes mid-statement. Patching `mil_driver.sh` while
   it ran killed it with `syntax error near unexpected token ')'` even though
   `bash -n` on the file passed. Write a new file and relaunch instead.

20. **`fit_unlearner.py` takes ONE `--encoder_dir` but "the rest" spans five
   cohort roots.** PROSTATE's negatives are TCGA / BRACS / BACH / UBC-OCEAN, each
   under its own root, so a single directory silently resolved zero of them:
   `X_pos: [64000, 2560], X_neg: [0]`. This matters even for plain SVD, whose
   subspace uses only `X_pos`, because `mu` is the mean pooled over BOTH cohorts.
   `--encoder_dir` now accepts a comma-separated list, tried per slide.
   Organ -> cohort: LUNG=TCGA, PROSTATE=PANDA, BREAST=BRACS(4539)/BACH(400)/TCGA(960).

21. **An eraser fitted with a truncated negative set is not comparable to one
   fitted properly.** The original `LUNG_k64.pt` used the master_benchmark root,
   so its "rest" was TCGA-BRCA alone. Its results are in
   `results/_superseded_mil/`. If you refit one organ's eraser, refit all of them.

22. **Never read scattered rows from these h5 files.** Features are chunked
   `(1, D)` on disk, so `feats[sorted_random_idx]` is one random chunk read per
   patch. On Lustre that ran at ~1 slide/s and killed two consecutive 55-minute
   eraser fits before a single one finished. Reading evenly spaced CONTIGUOUS
   slabs instead gives 0.62 s/slide for TCGA and 0.42 s/slide for the mixed
   negative set, a 1000-slide fit drops from >55 min to ~17 min, and keeps the
   sample spread across the slide. `quick_validate.slab_starts()` had already
   solved this; `fit_unlearner.py` had not, and the two were not sharing code.

23. **`pgrep -f <script>` matches the process doing the grepping.** A monitor
   whose own command line mentions `mil_driver2.sh` matched itself and reported
   the driver alive after it had exited. Same failure as the `pkill` trap that
   cost three attempts earlier. Use `ps -eo args | grep "[b]ash scripts/x.sh"`.

---

## Evaluation protocol (non-negotiable)

* Patient-grouped folds. Outer fold = **test**; early stopping watches an inner
  split carved from training patients. No max-over-epochs on the reported split.
* Probes retrained **from scratch** on the frozen eraser.
* **Both** probe families. `low_rank` scored 0.7301 on MLP (looked like erasure)
  while logreg recovered to 0.9369: above baseline. One probe family is not
  evidence.
* Report the high-capacity probe number alongside the standard one.

---

## Running things

```bash
python -m pytest tests/ -q                    # 26 tests, all should pass
bash scripts/smoke_test.sh                    # end-to-end, GPU node

# Fast validation on pooled features (minutes, not hours)
python scripts/quick_validate.py --target_task PANDA-grade \
    --control_task PANDA-detect --max_slides 600
```

**Why pooled features are a fair proxy:** every eraser here is linear and mean
pooling is linear, so `mean(P z_i) = P(mean z_i)`. A probe on erased slide-means
is *exactly* MeanMIL on erased patches, and close to ABMIL. Hours → minutes. It
does not capture attention re-weighting, so confirm headline claims with the full
MIL pipeline.

### Cluster notes (NCSA Delta)

* Batch GPU queues are thousands deep (`gpuA100x4`: 1543 pending). The
  **interactive** partitions are near-empty and schedule in seconds, capped at
  1 hour and **2 concurrent/submitted jobs per user** (`QOSMaxSubmitJobPerUserLimit`).
* Login node has a **30-minute CPU limit**: do not load features there.
* `/work/hdd` is heavily contended. Features are chunked `(1, 2560)`, so random
  row sampling means one seek per patch. Use **strided slab reads** (8 contiguous
  blocks of 32), this was ~9× faster. Feature caches are in
  `results/quick/cache/` and make reruns instant.
* Default partition is `gpuA40x4,gpuA100x4` (`scripts/common.sh`); override with
  `PURGE_SLURM_PARTITION`.

---

## Open questions

1. **Is plain SVD partly removing cohort structure wholesale?** Variance ranking
   beating relevance ranking hints that the top principal span describes "where
   this cohort lives", with task erasure as a consequence. The same-slides PANDA
   result argues against the pure version (detection survived at 0.79 while
   grading died at 0.48) but does not fully separate it. A reviewer will ask.

2. **The attack is not stealthy.** cos(z, z′) ≈ 0.86 after the affine fix (was
   0.43). Plan §23's "visually normal latent geometry" claim does not hold at that
   level: either weaken it or find distortion-aware selection that does not kill
   erasure (`λ_d` does not).

4. **Nested tasks.** Can grading be separated from detection at all, or is the
   overlap irreducible? Currently the clearest limitation.

5. Single fold, single encoder (virchow2), 600 slides/dataset. Nothing here is
   multi-fold yet.

---

## Repo state

* `results/_invalid_pre_fix/`: quarantined pre-fix results. Invalid (invertible
  eraser, wrong LEACE bias, selection on the eval split). Do not cite or
  aggregate. Kept for auditability.
* Patient arrays are now **sorted** before `KFold` for reproducibility. This
  changed fold membership: nothing from the quarantined runs is comparable to
  current numbers, including baselines.
* All current results: `results/quick/*.json`.
