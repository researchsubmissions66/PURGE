# Selective Representation Unlearning for Pathology Foundation Models
## Executable Experiment Ideas

### Goal

Study whether **post-hoc unlearning of selected information from already-extracted pathology foundation model features** can:

1. remove unwanted information from the representation,
2. preserve or improve downstream pathology performance,
3. reveal which factors positively or negatively transfer across tasks.

The key setting is:

```text
Pre-extracted patch features
        ↓
Selective representation unlearning
        ↓
Modified patch features
        ↓
ABMIL
        ↓
Downstream slide-level task
```

No foundation-model re-encoding is required for the core experiments.

---

# 1. Core Research Question

Given a frozen pathology foundation model feature extractor

\[
z = f_\theta(x),
\]

learn a post-hoc transformation

\[
T_\phi : z \rightarrow z'
\]

that suppresses a selected concept \(c\), while retaining information useful for target task \(y\).

Desired behavior:

\[
I(z'; c) \ll I(z;c)
\]

while

\[
P(y|z') \approx P(y|z)
\]

or ideally

\[
P(y|z') > P(y|z).
\]

Examples of concepts to unlearn:

- organ identity
- institution/site
- scanner
- stain/style
- cohort/dataset
- magnification
- normal-tissue architecture
- selected morphology concepts

---

# 2. First Executable Experiment

## Hypothesis

Removing kidney- and colon-specific information from a pan-organ pathology FM may improve or preserve lung subtype discrimination by reducing irrelevant representation structure.

### Forget concepts

```text
Kidney
Colon
Kidney + Colon
```

### Retain / downstream task

```text
TCGA-NSCLC
LUAD vs LUSC
ABMIL
```

### Conditions

| ID | Representation |
|---|---|
| E0 | Original features |
| E1 | Kidney-unlearned |
| E2 | Colon-unlearned |
| E3 | Kidney + Colon-unlearned |
| E4 | Random matched-dimension removal |
| E5 | Lung-unlearned |
| E6 | Full organ-invariant representation |

The strongest expected qualitative pattern would be:

```text
E3 > E0 ≈ E1/E2 > E4 >> E5
```

but **do not require E3 > E0** for the experiment to be useful.

A null result can still reveal that multi-organ information is shared rather than interfering.

---

# 3. Data Organization

Recommended input structure:

```text
features/
├── UNI/
│   ├── TCGA_LUNG/
│   │   ├── patient_001/
│   │   │   ├── slide_001.pt
│   │   │   └── slide_002.pt
│   │   └── ...
│   ├── TCGA_RCC/
│   ├── TCGA_COAD/
│   └── ...
├── Virchow2/
└── PhikonV2/
```

Each `.pt` file should contain approximately:

```python
{
    "features": Tensor[N_patches, D],
    "coords": Tensor[N_patches, 2],   # optional
    "slide_id": str,
    "patient_id": str,
    "label": int,
    "organ": str,
}
```

Do all train/validation/test splitting at the **patient level**.

Never allow patches from one patient to appear in multiple splits.

---

# 4. Minimum Metadata Table

Create a CSV like:

```text
slide_id,patient_id,organ,dataset,label,feature_path
TCGA-XX-0001,TCGA-XX,LUNG,TCGA,LUAD,/.../slide.pt
TCGA-YY-0002,TCGA-YY,LUNG,TCGA,LUSC,/.../slide.pt
...
```

Useful optional columns:

```text
institution
scanner
stain_batch
magnification
sex
age
site
cohort
```

These enable additional unlearning experiments later.

---

# 5. Experiment A — Verify Organ Information Exists

Before unlearning anything, measure whether organ identity is recoverable.

Pool patch embeddings from:

```text
Lung
Kidney
Colon
Breast
```

Train:

- logistic regression
- linear SVM
- 2-layer MLP
- kNN

### Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

clf = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    n_jobs=-1,
)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Macro-F1:", f1_score(y_test, pred, average="macro"))
```

If organ accuracy is high, the FM contains easily decodable organ information.

---

# 6. Experiment B — Simplest Unlearning Baseline

Start with a one-vs-rest linear organ direction.

Example:

```text
Kidney = 1
Everything else = 0
```

Fit logistic regression and obtain weight vector \(w\).

Remove its direction:

\[
z' =
z -
\frac{z^\top w}{w^\top w}w
\]

### PyTorch implementation

```python
import torch

def remove_direction(X, w, eps=1e-12):
    # X: [N, D]
    # w: [D]
    w = w.to(X.device, X.dtype)
    denom = torch.dot(w, w).clamp_min(eps)

    coeff = (X @ w) / denom
    X_new = X - coeff.unsqueeze(1) * w.unsqueeze(0)

    return X_new
```

This is not expected to fully erase an organ, but it is an excellent sanity-check baseline.

---

# 7. Experiment C — Multi-Dimensional Subspace Removal

An organ is unlikely to occupy one direction.

Learn an orthonormal basis:

\[
U_k =
[u_1,\ldots,u_k]
\]

and remove:

\[
z' = (I-U_kU_k^\top)z.
\]

### PyTorch implementation

```python
def remove_subspace(X, U):
    # X: [N, D]
    # U: [D, K], orthonormal columns
    return X - (X @ U) @ U.T
```

Try:

```text
K = 1
K = 2
K = 4
K = 8
K = 16
K = 32
K = 64
```

---

# 8. How to Estimate the Forget Subspace

## Option 1 — Classifier weights

Train multiple binary classifiers:

```text
Kidney vs Rest
Colon vs Rest
```

Stack weights:

\[
W = [w_K, w_C].
\]

Run QR:

```python
W = torch.stack([w_kidney, w_colon], dim=1)
U, _ = torch.linalg.qr(W)
```

Then remove the resulting subspace.

---

## Option 2 — Mean-difference directions

For organ \(c\):

\[
v_c =
\mu_c - \mu_{\neg c}.
\]

Implementation:

```python
def mean_difference_direction(X_pos, X_neg):
    v = X_pos.mean(0) - X_neg.mean(0)
    return v / v.norm()
```

This is extremely cheap and useful as a baseline.

---

## Option 3 — PCA on between-organ residuals

Compute organ centroids:

\[
\mu_o
\]

and perform PCA over organ-centered structure.

This provides several organ-associated directions rather than one.

---

## Option 4 — LEACE-style erasure

Use linear concept erasure to remove covariance between representation and concept labels while minimally distorting the embeddings.

Recommended as a serious baseline.

---

## Option 5 — SPLINCE-style task-aware projection

Preferable when you explicitly want:

```text
Forget: Kidney + Colon
Preserve: LUAD vs LUSC
```

Conceptually optimize a transformation that removes covariance with the forget concept while preserving covariance with the target task.

This should be one of the strongest baselines for the final paper.

---

# 9. Task-Aware Low-Rank Unlearning Method

A simple learnable method:

\[
z' = z + UV^\top z
\]

where

\[
U,V \in \mathbb{R}^{D \times r},
\qquad
r \ll D.
\]

For example:

```text
D = 1024
r = 16 or 32
```

Trainable parameters:

```text
2 × D × r
```

For:

```text
D = 1024
r = 32
```

only:

```text
65,536 parameters
```

---

# 10. Low-Rank Eraser Module

```python
import torch
import torch.nn as nn


class LowRankEraser(nn.Module):
    def __init__(self, dim, rank=32):
        super().__init__()

        self.U = nn.Parameter(
            torch.randn(dim, rank) * 1e-3
        )

        self.V = nn.Parameter(
            torch.randn(dim, rank) * 1e-3
        )

    def forward(self, x):
        delta = (x @ self.V) @ self.U.T
        return x + delta
```

---

# 11. Forget Adversary

Train a classifier to recover the unwanted concept.

```python
class ForgetClassifier(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)
```

Possible labels:

```text
0 = Other
1 = Kidney
2 = Colon
```

---

# 12. Retain Head

For a task-aware representation method, train a lightweight retain head.

For lung:

```text
0 = LUAD
1 = LUSC
```

```python
class RetainClassifier(nn.Module):
    def __init__(self, dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.fc(x)
```

Important:

Patch-level lung subtype labels are inherited from the slide and are weak labels.

For the final evaluation, ABMIL remains the authoritative downstream model.

---

# 13. Recommended Objective

Use:

\[
\mathcal L =
\lambda_f \mathcal L_{\text{forget}}
+
\lambda_r \mathcal L_{\text{retain}}
+
\lambda_d \mathcal L_{\text{distortion}}
\]

where:

### Forget term

Make the adversary perform poorly.

Possible implementation with gradient reversal:

\[
\mathcal L_{\text{forget}}
=
-\mathrm{CE}(g(z'),c)
\]

### Retain term

\[
\mathcal L_{\text{retain}}
=
\mathrm{CE}(h(z'),y)
\]

### Distortion term

\[
\mathcal L_{\text{distortion}}
=
\|z'-z\|_2^2.
\]

Optional neighborhood preservation:

\[
\mathcal L_{\text{geometry}}
=
\left\|
ZZ^\top -
Z'Z'^\top
\right\|_F^2.
\]

Final:

\[
\mathcal L =
\lambda_f L_f
+
\lambda_r L_r
+
\lambda_d L_d
+
\lambda_g L_g.
\]

Suggested starting values:

```text
lambda_forget = 1.0
lambda_retain = 1.0
lambda_distortion = 0.01
lambda_geometry = 0.001
```

Tune only after the pilot.

---

# 14. ABMIL Downstream Evaluation

For each representation condition, train a **fresh ABMIL model from scratch**.

Do not reuse ABMIL weights.

Pipeline:

```text
Original features
       ↓
ABMIL
       ↓
LUAD/LUSC

Kidney+Colon-unlearned features
       ↓
Fresh ABMIL
       ↓
LUAD/LUSC
```

The only variable should be the representation.

---

# 15. Minimal ABMIL

```python
import torch
import torch.nn as nn


class ABMIL(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        attention_dim=128,
        num_classes=2,
    ):
        super().__init__()

        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes,
        )

    def forward(self, x):
        # x: [N_patches, D]
        h = self.feature_proj(x)

        a = self.attention(h)
        a = torch.softmax(a.squeeze(-1), dim=0)

        bag = torch.sum(
            h * a.unsqueeze(-1),
            dim=0,
        )

        logits = self.classifier(bag)

        return logits, a
```

---

# 16. Pilot Evaluation Matrix

Run:

| Experiment | Forget target | Lung ABMIL |
|---|---|---|
| E0 | None | LUAD/LUSC |
| E1 | Kidney | LUAD/LUSC |
| E2 | Colon | LUAD/LUSC |
| E3 | Kidney + Colon | LUAD/LUSC |
| E4 | Random matched subspace | LUAD/LUSC |
| E5 | Lung | LUAD/LUSC |

Use:

```text
5 patient-level folds
```

Primary metric:

```text
AUROC
```

Secondary:

```text
Accuracy
Macro-F1
Balanced Accuracy
AUPRC
```

---

# 17. Critical Forgetting Evaluation

Downstream ABMIL performance alone does **not** prove successful unlearning.

After producing \(z'\), train completely fresh attackers:

```text
Logistic regression
Linear SVM
MLP
kNN
```

Try to recover:

```text
Kidney
Colon
Kidney vs Colon
Kidney/Colon vs Rest
```

Desired outcome:

```text
Original representation:
    high organ recovery

Unlearned representation:
    organ recovery ≈ chance
```

---

# 18. Random-Subspace Control

If you remove \(K\) dimensions, remove \(K\) random orthonormal dimensions as a control.

```python
def random_orthonormal_basis(dim, rank, device="cpu"):
    A = torch.randn(dim, rank, device=device)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :rank]
```

Repeat random control:

```text
20–50 times
```

Compare mean ± std.

This is essential because otherwise reviewers can argue that gains arise simply from dimensionality reduction.

---

# 19. Positive Control

Unlearn the target organ itself:

```text
Forget Lung
Evaluate LUAD vs LUSC
```

Expected:

\[
AUC_{-\text{lung}} < AUC_{\text{original}}.
\]

If lung unlearning has no effect at all, your unlearning transformation may not be removing meaningful biological information.

---

# 20. Few-Shot Experiment

This may be more promising than full-shot.

Use:

```text
4-shot
8-shot
16-shot
32-shot
Full-shot
```

per class.

Hypothesis:

\[
\Delta_{\text{unlearning}}
\]

may be larger when ABMIL has fewer training slides because irrelevant representation structure is harder to ignore under low-data conditions.

Recommended table:

| Features | 4-shot | 8-shot | 16-shot | 32-shot | Full |
|---|---:|---:|---:|---:|---:|
| Original | | | | | |
| - Kidney | | | | | |
| - Colon | | | | | |
| - Kidney-Colon | | | | | |
| Random | | | | | |
| - Lung | | | | | |

---

# 21. Organ Interaction Matrix

This could become a central scientific result.

Define:

\[
M_{ij}
=
P_j(E_{-i}) - P_j(E_0)
\]

where:

```text
i = organ being unlearned
j = downstream organ/task
```

Example:

| Unlearn ↓ / Evaluate → | Lung | Kidney | Colon | Breast |
|---|---:|---:|---:|---:|
| None | baseline | baseline | baseline | baseline |
| Lung | | | | |
| Kidney | | | | |
| Colon | | | | |
| Breast | | | | |

Interpretation:

```text
negative value:
    organ i contributed positively to task j

positive value:
    organ i interfered with task j

near zero:
    organ i is largely irrelevant to task j
```

This measures **cross-organ transfer/interference inside pathology foundation models**.

---

# 22. Better Nuisance-Unlearning Experiments

Organ unlearning is biologically ambiguous.

The following nuisance variables may yield cleaner results.

## Experiment N1 — Institution Unlearning

Forget:

```text
TCGA tissue source site / institution
```

Evaluate:

```text
LUAD vs LUSC ABMIL
```

Best evaluation:

```text
Train: TCGA-LUNG
Test: CPTAC-LUNG
```

## Experiment N2 — Dataset/Cohort Unlearning

Forget:

```text
TCGA vs CPTAC
```

while retaining:

```text
lung morphology
```

Evaluate:

```text
cross-cohort LUAD/LUSC
```

## Experiment N3 — Scanner Unlearning

Forget:

```text
scanner identity
```

Evaluate:

```text
train scanner A
test scanner B
```

## Experiment N4 — Stain Unlearning

Generate stain variants of the same patch.

Desired:

\[
T(f(x^{(1)}))
\approx
T(f(x^{(2)}))
\]

while preserving morphology.

## Experiment N5 — Magnification Unlearning

Forget:

```text
10x / 20x / 40x
```

while retaining tissue morphology.

---

# 23. Morphology-Concept Unlearning

Forget concepts such as:

```text
TIL
stroma
necrosis
glands
adipose
tumor
normal epithelium
```

Then test the effect on downstream tasks.

Example:

```text
Unlearn TIL
        ↓
MSI prediction
```

This turns unlearning into a tool for **representation interrogation**.

---

# 24. Geometry Analysis

For original and unlearned features, compute:

```text
CKA
SVCCA
PWCCA
Procrustes distance
effective rank
pairwise cosine similarity
kNN preservation
```

Questions:

```text
How much did the representation move?
Did unlearning alter only a small subspace?
Did local morphology neighborhoods survive?
Did global geometry collapse?
```

---

# 25. Attention Analysis

After ABMIL training, compare attention maps between:

```text
Original features
vs
Unlearned features
```

Measure:

```text
Spearman correlation
Jaccard overlap of top-k patches
KL divergence
attention entropy
```

This can reveal whether unlearning changes **which morphology ABMIL relies on**.

---

# 26. Representation-Retention Metrics

### Cosine retention

\[
R_{\cos}
=
\frac{1}{N}
\sum_i
\cos(z_i,z_i')
\]

### Neighborhood retention

Compare top-k neighbors before and after.

Measure:

```text
Recall@k
Jaccard@k
```

### Relative distortion

\[
D =
\frac{
\|Z'-Z\|_F
}{
\|Z\|_F
}
\]

Lower distortion is preferable when forgetting is equivalent.

---

# 27. Forget-Retain Frontier

For different removal strengths \(K\), plot:

```text
x-axis:
    residual forget-concept recoverability

y-axis:
    downstream ABMIL AUROC
```

Try:

```text
K = 1, 2, 4, 8, 16, 32, 64
```

A good method achieves:

```text
low forget recoverability
high downstream utility
```

---

# 28. Statistical Testing

Use paired evaluation because every representation uses the same folds.

Recommended:

```text
5 folds × multiple seeds
```

Report:

```text
mean ± standard deviation
95% bootstrap CI
```

For paired AUC comparisons:

```text
paired bootstrap
or
DeLong where appropriate
```

For fold-level comparisons:

```text
Wilcoxon signed-rank
```

Avoid overclaiming based on a 0.5–1% AUC difference with large variance.

---

# 29. Minimal Pilot Before Full Study

Run only:

```text
FM:
    UNI

Forget organs:
    Kidney
    Colon
    Kidney + Colon
    Lung

Downstream:
    TCGA LUAD vs LUSC

MIL:
    ABMIL

CV:
    5-fold patient-level

Methods:
    Original
    mean-direction removal
    classifier-direction removal
    random control
```

If there is a signal, add:

```text
LEACE
SPLINCE-style projection
low-rank task-aware eraser
additional PFMs
few-shot evaluation
external cohort testing
```

---

# 30. Suggested Repository Structure

```text
pathology-unlearning/
├── configs/
│   ├── uni_lung.yaml
│   ├── virchow2_lung.yaml
│   └── phikon_lung.yaml
│
├── data/
│   └── metadata.csv
│
├── src/
│   ├── datasets/
│   │   ├── feature_dataset.py
│   │   └── mil_dataset.py
│   │
│   ├── unlearning/
│   │   ├── direction.py
│   │   ├── subspace.py
│   │   ├── random_projection.py
│   │   ├── leace.py
│   │   ├── splince.py
│   │   └── low_rank_eraser.py
│   │
│   ├── models/
│   │   ├── abmil.py
│   │   ├── probes.py
│   │   └── adversary.py
│   │
│   ├── evaluation/
│   │   ├── forgetting.py
│   │   ├── downstream.py
│   │   ├── geometry.py
│   │   └── statistics.py
│   │
│   └── utils/
│       ├── splits.py
│       ├── seed.py
│       └── io.py
│
├── scripts/
│   ├── fit_unlearner.py
│   ├── transform_features.py
│   ├── train_abmil.py
│   ├── evaluate_forgetting.py
│   ├── evaluate_geometry.py
│   └── run_pilot.sh
│
└── results/
```

---

# 31. Example Configuration

```yaml
experiment:
  name: uni_kidney_colon_unlearning_lung

features:
  encoder: UNI
  dimension: 1024

forget:
  concepts:
    - kidney
    - colon

retain:
  organ: lung
  task: luad_vs_lusc

unlearning:
  method: subspace
  rank: 16

mil:
  model: abmil
  hidden_dim: 256
  attention_dim: 128
  dropout: 0.25

training:
  folds: 5
  seeds:
    - 1
    - 2
    - 3
  epochs: 50
  lr: 0.0001

evaluation:
  primary_metric: auroc
  metrics:
    - auroc
    - accuracy
    - macro_f1
    - auprc
```

---

# 32. Example Pilot Shell Script

```bash
#!/bin/bash

ENCODER="UNI"
TARGET="lung"

for FORGET in kidney colon kidney_colon lung random
do
    python scripts/fit_unlearner.py \
        --encoder $ENCODER \
        --forget $FORGET \
        --output results/unlearners/${ENCODER}_${FORGET}.pt

    python scripts/transform_features.py \
        --encoder $ENCODER \
        --unlearner results/unlearners/${ENCODER}_${FORGET}.pt \
        --organ $TARGET \
        --output results/features/${ENCODER}_${FORGET}_${TARGET}

    for FOLD in 0 1 2 3 4
    do
        python scripts/train_abmil.py \
            --features results/features/${ENCODER}_${FORGET}_${TARGET} \
            --fold $FOLD \
            --task luad_lusc
    done
done
```

---

# 33. Decision Criteria

Continue toward a full paper if at least one of the following happens.

### Outcome A

```text
Kidney/colon recoverability strongly decreases
AND
lung ABMIL stays unchanged
```

Interpretation:

```text
selective information removal is possible
```

### Outcome B

```text
Kidney/colon recoverability strongly decreases
AND
lung ABMIL improves
```

Interpretation:

```text
non-target representation structure interferes with adaptation
```

### Outcome C

```text
Kidney/colon removal hurts lung
```

Interpretation:

```text
cross-organ morphology is positively shared
```

### Outcome D

```text
Different organs cause different downstream changes
```

Interpretation:

```text
foundation models contain measurable cross-organ transfer structure
```

### Outcome E

```text
nuisance unlearning improves TCGA → CPTAC transfer
```

Interpretation:

```text
post-hoc representation unlearning improves domain robustness
```

---

# 34. Recommended Priority Order

## Phase 1 — Fast sanity check

```text
UNI
Kidney / Colon / Lung
simple linear directions
TCGA-LUNG
ABMIL
5-fold CV
```

## Phase 2 — Serious baseline study

```text
Original
Random
LEACE
SPLINCE
low-rank eraser
```

## Phase 3 — Generalization

```text
UNI
Virchow2
Phikon-v2
MUSK
```

## Phase 4 — Few-shot

```text
4 / 8 / 16 / 32-shot
```

## Phase 5 — External validation

```text
TCGA → CPTAC
```

## Phase 6 — Broader unlearning

```text
institution
dataset
scanner
stain
magnification
organ
```

---

# 35. Strongest Paper-Level Framing

Possible framing:

> **Task-Aware Representation Unlearning for Pathology Foundation Models**

Core question:

> Can unwanted information be selectively removed from frozen pathology foundation model embeddings without sacrificing—and potentially improving—downstream clinical utility?

Alternative framing:

> **Representation Specialization through Selective Unlearning in Pathology Foundation Models**

Broader scientific question:

> Which components of pan-organ pathology foundation-model representations are task-relevant, transferable, interfering, or purely technical?

---

# 36. Most Important Experimental Rule

Do **not** claim successful unlearning based only on downstream performance.

Every experiment should contain:

```text
1. Forgetting test
2. Retention/downstream test
3. Random-control test
4. Target-concept positive control
```

For the kidney + colon → lung example:

```text
FORGETTING
    Can a fresh model recover kidney/colon?

RETENTION
    Can ABMIL still classify LUAD/LUSC?

RANDOM CONTROL
    What happens if the same number of random directions are removed?

POSITIVE CONTROL
    What happens if lung itself is removed?
```

That four-part structure should make the experimental claims much harder to challenge.
