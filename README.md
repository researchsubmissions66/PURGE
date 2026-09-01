<p align="center">
  <img src="docs/assets/logo.png" alt="PURGE Logo" width="500">
</p>

<h1 align="center">Concept Poisoning in the Latent Space:<br>Pathology Unlearning via Representational Geometry Erasure</h1>

<p align="center">
  <a href="https://researchsubmissions66.github.io/PURGE/"><img src="https://img.shields.io/badge/Project-Website-6b21a8?style=for-the-badge" alt="Website"></a>
  <a href="#"><img src="https://img.shields.io/badge/Paper-Coming%20Soon-1e1b4b?style=for-the-badge" alt="Paper"></a>
</p>

---

## Overview

**PURGE** studies *targeted adversarial attacks on representations*: can a specific
clinical concept be made unrecoverable from frozen pathology foundation-model
embeddings, while unrelated downstream tasks and the latent geometry survive?

The attack learns a transform `z' = A(z)` over patch embeddings such that a probe
**retrained from scratch** on `z'` cannot recover the target task, while probes for
control tasks are unaffected.

## The one rule that governs this repo

**An eraser must be non-invertible.** An invertible map — including the residual
adapter `z' = z + αBAz` — is an information-preserving bijection. It can fool the
fixed classifier it was trained against, but a retrained probe recovers the target
concept exactly. That is the trivial attack, not erasure.

Accordingly:

* The primary attack is affine SVD null-space projection
  (`src/unlearning/subspace.py`), which removes `k` directions and annihilates
  them.
* `src/unlearning/audit.py` inspects any eraser and reports whether it can destroy
  information at all. Run it before trusting any new eraser.
* `LowRankEraser` is retained only as a documented negative control.
* **Only AUCs from probes retrained from scratch are results.** The co-adapted
  in-loop adversary always looks better than reality; it is printed labelled
  `(NOT a result)`.

## Evaluation protocol

* Patient-grouped 5-fold CV. The outer fold is the **test** set; early stopping
  watches an inner validation split carved from the training patients, so the
  reported number is not selected on the split it is measured on.
* Every AUC is macro one-vs-rest. **Chance is 0.50 regardless of class count** — a
  successful erasure drives the target to ~0.50, not to `1/num_classes`.
* An undefined AUC is reported as `null` with a reason, never as a placeholder
  `0.5` (which would be indistinguishable from a genuine chance-level result).

## Status of results

The results previously published on the project site were produced with an
invertible eraser, an incorrectly applied LEACE bias term, and max-over-epochs
selection on the evaluation split. **They are being regenerated under the corrected
pipeline and should not be cited.** What currently holds:

| Finding | Status |
|---------|--------|
| Residual low-rank adapters cannot erase | Proven analytically + pinned by a test |
| Gaussian noise / feature dropout do not selectively degrade | Supported |
| Linear concept erasure (LEACE/SPLINCE) vs. MIL probes | Re-running — earlier runs used a wrong bias term |
| Learned null-space projection attack | Re-running |

## Datasets

| Dataset | Organ | Task | Classes |
|---------|-------|------|---------|
| **PANDA** | Prostate | ISUP grading | 6 |
| **BACH** | Breast | Histology classification | 4 |
| **BRACS** | Breast | Lesion subtyping | 7 |
| **UBC-OCEAN** | Ovarian | Carcinoma subtyping | 5 |
| **TCGA-LUNG** | Lung | LUAD vs. LUSC | 2 |
| **TCGA-BRCA** | Breast | IDC vs. ILC | 2 |

Note that BACH, BRACS and TCGA-BRCA are all breast: a BACH→BRACS pair is a
*same-organ* control, not a cross-organ one.

## Foundation models

Swept by the launchers (`scripts/common.sh`): `hoptimus0`, `virchow`, `virchow2`,
`gpfm`, `uni_v2`, `gigapath`, `ctranspath`, `conch_v15`. Embedding dimensionality is
read from the features at runtime — nothing is hardcoded.

## Erasure methods

| Method | `--unlearn_method` | Non-invertible? |
|--------|--------------------|-----------------|
| **SVD subspace removal (affine)** | `svd` | Yes (rank `d − k`) |
| LEACE | `leace` | Yes |
| SPLINCE | `splince` | Yes |
| Residual low-rank adapter | `low_rank` | **No** — negative control only |
| Gaussian noise | `gaussian` | No — baseline |
| Feature dropout | `dropout` | No — baseline |

Eight attempts to improve on affine SVD were measured and removed: adversarial
learned erasers, relevance re-ranking (Fisher and HSIC), iterative INLP, a
reconstruction head, background whitening, automatic rank selection and bootstrap
stabilisation. They all lose to plain variance ranking, for one reason: task
information is distributed across the cohort's principal span rather than
concentrated in directions you can select cleverly. The numbers and the mechanism
are in `AGENTS.md`; the code is gone so nobody rebuilds it by accident.

### The default eraser: affine SVD

`--unlearn_method svd` applies the **mean-preserving** form

```
z' = (z - mu) - ((z - mu) @ U) @ U.T + mu
```

not the plain `z @ (I - U Uᵀ)`. Both delete exactly the same information — the
discriminative signal lives in the centred data — but the plain form also deletes
the embedding mean's component along `U`. Foundation-model embeddings concentrate
most of their norm in a few dominant directions, so that costs a great deal of
fidelity for nothing:

| | erasure AUC | cos(z, z′) |
|---|---|---|
| plain, BACH | 0.4522 | 0.4300 |
| **affine, BACH** | **0.4521** | **0.8221** |
| plain, TCGA-LUNG | 0.4661 | 0.5897 |
| **affine, TCGA-LUNG** | **0.4661** | **0.8606** |

Identical erasure, roughly half the distortion. `--no_svd_affine` restores the
plain form for ablations only. SVD erasers are stored as `{'U', 'mu'}`; a legacy
bare-tensor checkpoint still loads but warns and falls back to the plain form.

## Repository structure

```
PURGE/
├── configs/
├── data/                              # metadata CSVs (features live on scratch)
├── docs/                              # project website
├── src/
│   ├── datasets/feature_dataset.py    # HDF5 loading; benchmark-wide label map
│   ├── evaluation/metrics.py          # macro OVR AUC, selective degradation score
│   ├── models/                        # ABMIL / MeanMIL / TransMIL + factory
│   ├── unlearning/
│   │   ├── subspace.py                # affine SVD  <- the attack
│   │   ├── spectral.py                # control-aware generalisation (lambda=0 == SVD)
│   │   ├── lora.py                    # adapters for encoder-side poisoning
│   │   ├── low_rank.py                # invertible adapter (negative control)
│   │   ├── losses.py                  # preservation, adversarial, HSIC
│   │   ├── apply.py                   # build/save/load/apply every eraser
│   │   ├── audit.py                   # is this eraser capable of erasing?
│   │   ├── noise.py                   # noise & dropout baselines
│   │   ├── concept_erasure/           # vendored LEACE (EleutherAI, MIT)
│   │   └── splince/                   # vendored SPLINCE
│   └── utils/splits.py                # patient-grouped folds + inner val split
├── scripts/
│   ├── train_mil.py                   # train/evaluate a probe through an eraser
│   ├── quick_validate.py              # fast pooled-feature validation
│   ├── poison_encoder.py              # encoder-side poisoning (plan section 19)
│   ├── sanity_check_erasure.py        # falsification suite
│   ├── fit_unlearner.py               # closed-form erasers (svd / leace / splince)
│   ├── linear_probe.py                # fast linear recoverability check
│   ├── aggregate_results.py           # baseline table + target x control matrix
│   ├── stat_test.py                   # paired significance tests
│   ├── common.sh                      # shared SLURM + path configuration
│   ├── launch_{baselines,erasers,evaluations}.sh
│   └── smoke_test.sh                  # end-to-end check before a sweep
└── tests/test_erasers.py
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -q

# 1. metadata
python scripts/fetch_tcga_metadata.py
python scripts/build_multi_dataset.py

# 2. end-to-end check on one dataset pair
bash scripts/smoke_test.sh

# 3. full sweep
bash scripts/launch_baselines.sh     # reference AUCs
bash scripts/launch_erasers.sh       # fit one eraser per target
bash scripts/launch_evaluations.sh   # retrain fresh probes through each eraser
python scripts/aggregate_results.py
python scripts/stat_test.py
```

Fitting a single eraser directly:

```bash
python scripts/fit_unlearner.py \
    --forget_organ LUNG --encoder_dir /path/to/features_virchow2 \
    --method svd --k 64 --output results/unlearners/LUNG_k64.pt
```

## Citation

```bibtex
@article{purge2025,
  title={Concept Poisoning in the Latent Space: Pathology Unlearning via Representational Geometry Erasure},
  author={Anonymous},
  year={2025},
  note={Under review}
}
```
