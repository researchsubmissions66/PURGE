# AGENTS.md — PURGE

Context for anyone (human or agent) picking up this repo. Written after a full
validation pass. Read the traps section before running anything.

---

## What this project is

A **targeted adversarial attack on representations**: delete one downstream
clinical task from frozen pathology foundation-model embeddings while leaving
unrelated tasks intact. Not organ removal — the goal is task-specific erasure for
a specific organ.

Data is pre-extracted patch features in HDF5 (`/work/hdd/bhwm/...`), d=2560 for
`features_virchow2`. Slide-level MIL probes downstream.

---

## The one rule

**An eraser must be non-invertible.** An invertible map — including the residual
adapter `z' = z + αBAz` — is an information-preserving bijection. It fools the
fixed probe it was trained against; a *retrained* probe recovers the concept
exactly.

This is not theoretical. The original pipeline used exactly that adapter and
reported erasure that did not exist: measured `σ_min = 2.27e-02`, `rank 2560/2560`
— zero dimensions annihilated, and the eval AUC confirmed it (BACH→BACH 0.9884).

Guardrails now in place:

* `src/unlearning/audit.py` — `audit_eraser()` reports rank / invertibility.
* `scripts/train_adversarial.py` refuses to launch an invertible eraser without
  `--allow_invertible`.
* `tests/test_erasers.py::test_low_rank_adapter_is_invertible` pins the negative
  result so it cannot silently return.

**Never quote an AUC from a probe that was trained alongside the eraser.** The
co-adapted in-loop probe showed 0.58 while a retrained probe showed 0.99 on the
same eraser. Only from-scratch retrained probes count.

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

\* effectively clamped — see traps.

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

## Sanity checks — all passed

`scripts/sanity_check_erasure.py`. Each is built to fail if the result is an
artifact.

| Check | Result |
|---|---|
| Random subspace, same rank | 0.9404 vs baseline 0.9363 — **no effect** |
| Wrong cohort's subspace | 0.9341 vs 0.9363 — **no effect** |
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

### 1. Adversarial learned eraser — abandoned

`scripts/train_adversarial.py` with `ProjectionEraser`. The min-max does not
converge; in-loop probe accuracy climbs to 0.988 while the adversarial loss rises.

Initializing from the SVD subspace does not save it — **it starts on the answer
and walks away**:

| Target | svd_k64 drop | projection_svdinit_r64 drop |
|---|---|---|
| PANDA | +0.2399 | −0.0190 |
| BACH | +0.3605 | −0.0184 |
| UBC-OCEAN | +0.3530 | +0.0523 |

Code is kept and working; the approach is not recommended.

### 2. Relevance-weighted direction selection — a trade-off knob, not an upgrade

`src/unlearning/relevance.py`. Ranks directions by target relevance per unit of
(control damage + variance removed). 2×2 ablation, PANDA grade/detect, MLP:

| config | target ↓ | detection ↑ |
|---|---|---|
| plain SVD | **0.4832** | 0.7861 |
| relevance λc=0 λd=0 | 0.6646 | 0.7532 |
| relevance λc=0 **λd=1** | 0.8579 | 0.9050 |
| relevance **λc=1** λd=0 | 0.6465 | **0.8474** |

* `λ_c` **works** — +9.4 points of control preservation, mechanism confirmed.
* `λ_d` is **fatal** — erasure goes to zero. The target signal lives in the
  high-variance directions, so penalising them removes the attack. **Default is
  now 0.**
* Neither rescues the core deficit: ranking directions *individually* erases less
  than taking the cohort's principal span. Fisher scores each direction in
  isolation, so signal spread across individually-mediocre directions is invisible.

Swapping Fisher for HSIC (`--relevance_criterion hsic`) does **not** help
(0.7519 vs Fisher's 0.7276). The criterion's linearity was not the cause.

Keep it as an ablation row: it answers "why not pick directions more cleverly?"

### 3. Reconstruction head (distortion masking) — failed, but informative

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

### 4. Iterative INLP — failed

`src/unlearning/iterative.py`. Remove the directions a probe uses, refit, repeat.
At equal rank 64: TCGA-LUNG 0.7837 vs SVD's 0.4661; PANDA 0.8181 vs 0.4832.

### 5. SVD refinements — one win, two neutral/negative

`src/unlearning/svd_plus.py`.

| refinement | axis | verdict |
|---|---|---|
| affine (mean-preserving) | how removal is applied | **WIN — now the default** |
| spectral pencil (lambda) | control-aware variance | best collateral knob; costs erasure |
| bootstrap stabilisation | estimation noise | neutral (0.4661 -> 0.5056 mlp) |
| automatic k (parallel analysis) | the k knob | **worse** — picks k=19-35, erasure drops to 0.6341 |
| background whitening | the metric | **failed** — no erasure at all (0.9309) |

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

### 6. Spectral pencil — the cleanest knob, still not a free lunch

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

### 7. LEACE — exact linearly, leaks nonlinearly

Hits **exactly 0.5000** on a logistic probe every time (its guarantee, and it
holds once the bias is applied correctly). Against an MLP: 0.6823 / **0.9638** /
0.7299 / 0.8431 — for BACH it erases nothing. Correct statement is "linear
erasure does exactly what it promises and nothing more," not "linear erasure
fails."

---

## The central negative result

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
   Do **not** fix this by loosening the threshold — an invertible low-rank adapter
   has condition number ~3e4 and a loose cutoff would call it rank-deficient,
   defeating the audit.

5. **Two feature-cache formats exist.** Older `.npz` store `y` + `classes`; newer
   store raw `labels`. Loaders handle both; new code must too.

6. **Chance is 0.50 for macro one-vs-rest AUC**, regardless of class count. Not
   `1/num_classes`.

7. **Undefined AUC must never return 0.5.** A placeholder is indistinguishable
   from a genuine chance result — the exact quantity this project measures.
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

12. **BRACS-atypia is underpowered** — 167 cached slides of 4,539 available, ~133
   training. Every method is unstable there (0.90 → 0.20 → 0.47 across k). Prefer
   the PANDA pair (~470 training slides) for same-slides claims.

---

## Evaluation protocol (non-negotiable)

* Patient-grouped folds. Outer fold = **test**; early stopping watches an inner
  split carved from training patients. No max-over-epochs on the reported split.
* Probes retrained **from scratch** on the frozen eraser.
* **Both** probe families. `low_rank` scored 0.7301 on MLP (looked like erasure)
  while logreg recovered to 0.9369 — above baseline. One probe family is not
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
* Login node has a **30-minute CPU limit** — do not load features there.
* `/work/hdd` is heavily contended. Features are chunked `(1, 2560)`, so random
  row sampling means one seek per patch. Use **strided slab reads** (8 contiguous
  blocks of 32) — this was ~9× faster. Feature caches are in
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
   level — either weaken it or find distortion-aware selection that does not kill
   erasure (`λ_d` does not).

4. **Nested tasks.** Can grading be separated from detection at all, or is the
   overlap irreducible? Currently the clearest limitation.

5. Single fold, single encoder (virchow2), 600 slides/dataset. Nothing here is
   multi-fold yet.

---

## Repo state

* `results/_invalid_pre_fix/` — quarantined pre-fix results. Invalid (invertible
  eraser, wrong LEACE bias, selection on the eval split). Do not cite or
  aggregate. Kept for auditability.
* Patient arrays are now **sorted** before `KFold` for reproducibility. This
  changed fold membership — nothing from the quarantined runs is comparable to
  current numbers, including baselines.
* All current results: `results/quick/*.json`.
