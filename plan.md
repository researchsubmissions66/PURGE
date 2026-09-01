# Concept-Targeted Latent Space Poisoning in Computational Pathology

## 1. Motivation

Pathology foundation models (PFMs) such as UNI, Virchow, Phikon, CONCH, and MUSK generate rich latent representations that support a wide range of downstream tasks. A natural security and representation-learning question is:

> Can we selectively degrade the information required for one pathology task while preserving the usefulness of the embedding for unrelated tasks?

Rather than simply causing a fixed classifier to fail, the stronger goal is to make the target information itself difficult to recover from the embedding.

This can be viewed as task-selective representation erasure or, under an adversarial framing, concept-targeted latent space poisoning.

## 2. Problem Formulation

Let a pathology foundation model encoder be

$$z = E_\theta(x), \qquad z \in \mathbb{R}^{d},$$

where:
* $x$ is a pathology image patch,
* $E_\theta$ is a frozen pathology foundation model,
* $z$ is the extracted embedding.

Suppose the target task is $T$, with labels $Y_T$, for example:
* MSI status,
* tumor subtype,
* mutation status,
* grade,
* TIL abundance,
* immune phenotype,
* treatment response.

We want to construct a transformed embedding

$$z' = A_\phi(z)$$

such that information about the target task is strongly reduced:

$$I(z';Y_T) \downarrow,$$

while preserving information useful for unrelated tasks:

$$I(z';Y_{\text{other}}) \approx I(z;Y_{\text{other}}).$$

Ideally, the transformed representation should also remain close to the original representation:

$$z' \approx z.$$

The key objective is therefore: **Destroy target-task information while minimizing collateral damage to the latent representation.**

## 3. Why Fooling a Fixed Classifier Is Not Enough

A naive attack might optimize

$$\max_\phi CE(C_T(A_\phi(z)),y_T)$$

against a fixed classifier $C_T$.

This is insufficient. The transformation could simply reverse or distort the existing decision boundary while leaving the target information fully recoverable. A newly trained classifier might immediately recover the original performance.

For example:

* **Before:** MSI-H  --> classifier predicts MSI-H | MSI-L  --> classifier predicts MSI-L
* **After naive attack:** MSI-H  --> classifier predicts MSI-L | MSI-L  --> classifier predicts MSI-H

The original classifier fails, but the information still exists. A stronger requirement is: **A newly trained predictor should also fail to recover the target variable from the transformed embedding.** Therefore evaluation must always retrain probes on $z'$.

## 4. Baseline 1: Linear Concept Erasure

The easiest starting point is to identify a task-relevant linear subspace and project it out.

Let a linear target classifier be

$$p(y_T|z)=\operatorname{softmax}(Wz+b).$$

Suppose $U$ contains directions associated with the target concept. Then define

$$z' = (I-UU^\top)z.$$

This removes the task-relevant linear subspace.

Useful baselines include:
* linear projection,
* INLP-style iterative null-space projection,
* LEACE-style linear concept erasure.

### Evaluation
1. Extract original PFM embeddings.
2. Train the target classifier.
3. Compute the erasure transformation.
4. Transform all embeddings.
5. Discard the original classifier.
6. Train new classifiers from scratch on $z'$.
7. Compare target and control-task performance.

**Example:**

| Embedding | MSI AUC | Cancer Type | Tissue Type | Grade |
| :--- | :--- | :--- | :--- | :--- |
| Original UNI | 0.87 | 0.94 | 0.96 | 0.84 |
| Erased UNI | 0.53 | 0.93 | 0.95 | 0.83 |

A result like this would indicate selective target-task degradation.

## 5. Baseline 2: Learnable Low-Rank Task Eraser

A stronger approach is to learn a lightweight transformation over the frozen embeddings.

Use a residual low-rank adapter:

$$\boxed{z' = z + \alpha BA z}$$

where $A\in\mathbb{R}^{r\times d},\qquad B\in\mathbb{R}^{d\times r},$ and $r \ll d.$

Typical ranks might be: $r \in \{8,16,32,64\}.$

**Architecture:**
```
Histology Patch
      |
      v
+-------------+
| Frozen PFM  |
| UNI / etc.  |
+-------------+
      |
      v
      z
      |
      v
+------------------+
| Low-Rank Eraser  |
| z' = z + αBAz    |
+------------------+
      |
      v
      z'
```
Only the eraser parameters are trained initially.

## 6. Adversarial Target Erasure

Attach a target classifier $C_\psi(z') \rightarrow Y_T.$

The classifier attempts to predict the target: $CE(C_\psi(z'),Y_T).$
The eraser attempts to make the prediction difficult: $\max_\phi\mathcal{L}_T.$

This creates the min-max objective:

$$\min_\psi \max_\phi CE(C_\psi(A_\phi(z)),Y_T).$$

This can be implemented using alternating optimization, gradient reversal, or adversarial probe training.

## 7. Preventing the Trivial Solution

Without constraints, the eraser could simply collapse the representation: $z' = 0.$ This would destroy the target task, but also every other task. Therefore the attack needs preservation objectives.

A useful general objective is

$$-\lambda_T\mathcal L_{\text{target}}+\lambda_g\mathcal L_{\text{geometry}}+\lambda_c\mathcal L_{\cos}+\lambda_u\mathcal L_{\text{utility}}$$

where each term serves a different purpose.

## 8. Embedding Preservation Loss

Keep each transformed embedding close to its original version.

A simple cosine loss is $1-\cos(z,z').$
An alternative is Euclidean preservation: $\|z-z'\|_2^2.$

This discourages large global perturbations.

## 9. Latent Geometry Preservation

Preserve the pairwise structure of the embedding manifold.

Let $S_{ij}=z_i^\top z_j$ and $S'_{ij}=z_i'^\top z_j'.$

Then minimize $\|S-S'\|_F^2.$

Possible alternatives include preserving:
* cosine similarity matrices,
* pairwise Euclidean distances,
* local neighborhood structure,
* kNN graphs,
* covariance structure,
* principal subspaces.

This encourages the poisoned representation to remain globally similar to the original latent space.

## 10. Preserve Unrelated Tasks

Suppose we have $K$ control tasks $Y_1,\ldots,Y_K.$ For each control task, train or freeze a predictor $C_k$. Then define

$$\sum_{k=1}^{K}CE(C_k(z'),Y_k).$$

This directly penalizes collateral degradation. For example, if the target is MSI prediction, possible controls might include: tissue type, tumor vs. normal, cancer subtype, grade, organ identity, morphology classes.

The desired behavior is $\Delta P_{\text{target}} \gg\Delta P_{\text{control}}.$

## 11. Statistical Dependence Erasure

Instead of relying entirely on a target classifier, one can directly minimize dependence between the transformed embedding and the target variable.

For example:

$$\boxed{\min_\phi HSIC(Z',Y_T)}$$

subject to $D(Z,Z') < \epsilon.$

A practical loss is $\lambda_T HSIC(Z',Y_T)+\lambda_P D(Z,Z').$

This reframes the objective from "fool a classifier" to "eliminate statistical dependence between the representation and the target."

Possible dependence measures include: HSIC, mutual-information estimators, distance correlation, adversarial predictability.

## 12. Whole-Slide Image Formulation

For a whole slide $s$, let the PFM produce patch embeddings $Z_s =\{z_{s1},z_{s2},\ldots,z_{sN}\}.$

Apply the eraser independently: $z'_{si}=A_\phi(z_{si}).$

The transformed bag becomes $Z'_s =\{z'_{s1},z'_{s2},\ldots,z'_{sN}\}.$

Then use a MIL model: $\hat y_s =MIL(Z'_s).$

**Architecture:**
```
WSI
 |
 +--> Patch 1 --> PFM --> z1 --> Eraser --> z1'
 |
 +--> Patch 2 --> PFM --> z2 --> Eraser --> z2'
 |
 +--> ...
 |
 +--> Patch N --> PFM --> zN --> Eraser --> zN'
                                |
                                v
                         +--------------+
                         | MIL Model    |
                         | ABMIL/CLAM   |
                         | TransMIL/... |
                         +--------------+
                                |
                                v
                         Target Prediction
```

The target loss can be computed at the slide level: $CE(MIL(Z'_s),y_s).$

At the same time, representation preservation can be enforced at the patch level: $\frac{1}{N}\sum_i\left[1-\cos(z_{si},z'_{si})\right].$

This creates an interesting research question: *How little must patch-level latent geometry change before a clinically relevant slide-level phenotype becomes unrecoverable?*

## 13. Recommended First Experiment

A simple first setup would be:

* **Foundation model**: UNI
* **Target task**: Choose one slide-level task (e.g., MSI prediction, tumor subtype prediction, mutation prediction, survival-related phenotype).
* **Representation**: Pre-extracted patch embeddings from the frozen encoder.
* **Eraser**: Residual low-rank adapter: $z'=z+\alpha BA z.$
* **Suggested starting configuration**: rank ($r=16$), frozen PFM, trainable eraser only, cosine preservation, pairwise-geometry preservation, adversarial target probe.

**Architecture:**
```
                   +-------------------+
H&E Patches ------>| Frozen Pathology  |
                   | Foundation Model  |
                   +---------+---------+
                             |
                             v
                             z
                             |
                   +---------v---------+
                   | Low-Rank Task     |
                   | Eraser            |
                   +---------+---------+
                             |
                             v
                             z'
              +--------------+--------------+
              |              |              |
              v              v              v
        Target Probe     Control Tasks    Geometry
        Adversarial      Preservation     Preservation
```

## 14. Training Procedure

A practical training loop can alternate between two steps.

**Step A: Train the target adversary**
Freeze the eraser and optimize: $\psi\leftarrow\arg\min_\psi CE(C_\psi(A_\phi(z)),Y_T).$
The target adversary should always be strong enough to recover whatever information remains.

**Step B: Train the eraser**
Freeze the adversary and optimize: $\phi\leftarrow\arg\min_\phi\left[-\lambda_T\mathcal L_T+\lambda_P\mathcal L_{\text{preserve}}\right].$

Repeat these steps. An alternative is a gradient-reversal implementation.

## 15. Evaluation Protocol

Evaluation should explicitly test whether target information remains recoverable.

After training the eraser:
1. Freeze the final transformation.
2. Recompute all transformed embeddings.
3. Discard every classifier used during attack training.
4. Train entirely new probes on $z'$.

Use multiple probe families: logistic regression, linear SVM, RBF SVM, kNN, MLP, MIL models for slide-level tasks.

If only the original classifier fails, the result is weak. If multiple newly trained probes fail, the evidence for genuine target-information erasure is much stronger.

## 16. Metrics

* **Target degradation**: $P_T(E)-P_T(E').$
* **Collateral degradation**: $\frac{1}{K}\sum_{k=1}^{K}\left[P_k(E)-P_k(E')\right].$
* **Representation distortion**: Possible measures: $1-\cos(z,z'),$ $\|z-z'\|_2,$ or geometry-level distortion $\|S-S'\|_F.$

## 17. Selective Degradation Score

A useful summary metric could be

$$\frac{\Delta P_{\text{target}}}{\epsilon+\frac{1}{K}\sum_k |\Delta P_k|}$$

where:
* $\Delta P_{\text{target}}$ is the drop on the targeted task,
* $\Delta P_k$ is the drop on control task $k$,
* $\epsilon$ prevents division by zero.

A high SDS indicates strong degradation of the target task with minimal collateral degradation.

## 18. Desired Result

An ideal outcome might look like:

| Metric | Original | Poisoned |
| :--- | :--- | :--- |
| MSI AUC | 0.89 | 0.54 |
| Cancer subtype AUC | 0.93 | 0.92 |
| Tissue classification | 0.96 | 0.95 |
| Grade prediction | 0.85 | 0.84 |
| Mean cosine similarity | — | 0.98 |

This would demonstrate that a relatively small latent-space modification can selectively destroy one clinically meaningful signal.

## 19. From External Eraser to True PFM Poisoning

The first experiments should operate on frozen embeddings because they are cheaper, easier to debug, easier to analyze, and easier to compare across models.

Once the external eraser works, move the attack inside the foundation model.
Possible options:
* **LoRA**: Insert low-rank adapters into the final transformer blocks.
* **Partial Fine-Tuning**: Fine-tune only the last 1--4 transformer blocks.
* **Prompt/Token Adaptation**: For models that support learned tokens or multimodal prompts, optimize these components while freezing the backbone.

Then the modified foundation model itself produces $E_{\theta'}(x)=z'$ without requiring an external eraser. This is a stronger form of latent-space poisoning.

## 20. Experimental Matrix

A paper-scale study could evaluate:

* **Foundation Models**: UNI, Virchow / Virchow2, Phikon / Phikon-v2, CONCH, MUSK
* **Target Tasks**: MSI, cancer subtype, mutation status, grade, immune phenotype, treatment response
* **Erasure Methods**: linear projection, INLP, LEACE, low-rank adversarial eraser, HSIC-based erasure, LoRA-based internal poisoning.
* **Downstream Probes**: logistic regression, SVM, MLP, kNN, ABMIL, CLAM, TransMIL.

## 21. Important Ablations

* **Rank**: $r \in \{4,8,16,32,64,128\}.$
* **Preservation Strength**: Sweep $\lambda_P.$ This produces a trade-off between target destruction and representation fidelity.
* **Number of Target Samples**: Evaluate how much labeled data is required to erase a task.
* **Attack Transferability**: Train the eraser using one downstream architecture and evaluate against another.
* **Cross-Dataset Transfer**: Learn an eraser on one cohort and test whether target degradation transfers to another cohort.

## 22. Stronger Research Questions

The method enables several interesting questions.
* **How concentrated is task information?** If a rank-8 transformation removes most MSI information, this suggests MSI-relevant information occupies a small latent subspace.
* **Are some PFMs easier to selectively erase?** Different pathology foundation models may encode biological concepts with different degrees of redundancy.
* **Does representation size matter?** Larger representations may contain more redundant target information and therefore require higher-rank transformations.
* **Are clinically important concepts more robust?** Mutation, morphology, immune phenotype, and tissue identity may have very different erasure difficulty.
* **Can standard benchmarks detect the poisoned encoder?** A particularly important result would be an encoder that performs normally on common benchmarks while failing on one hidden clinically meaningful capability.

## 23. Potential Security Framing

The attack can be framed as a supply-chain threat. An attacker releases a seemingly normal pathology foundation model.

The model:
* performs well on standard benchmarks,
* produces visually normal latent geometry,
* retains most downstream capabilities,
* but one specific phenotype has been deliberately removed or corrupted.

For example:
```
Released PFM
 |
 +-- Cancer classification ........ Normal
 +-- Tissue classification ........ Normal
 +-- Grade prediction ............. Normal
 +-- MSI prediction ............... Broken
```
The malicious modification may therefore remain undetected under standard evaluation. This motivates both concept-targeted representation attacks, and auditing methods for hidden capability degradation in pathology foundation models.

## 24. Minimal Viable Study

A compact first study could be:
* **Model**: UNI
* **Dataset**: One cohort with slide-level MSI labels.
* **Representation**: Frozen patch embeddings.
* **Target**: MSI.
* **Controls**: cancer subtype, tissue morphology, tumor/stroma, one unrelated slide-level task.
* **Methods**: Original embeddings, Linear projection, LEACE/INLP, Rank-16 adversarial eraser, Rank-16 eraser + geometry preservation.
* **Evaluation**: linear probe, MLP probe, ABMIL, representation similarity, control-task performance.

If selective degradation already appears here, the idea is worth scaling.

## 25. Core Hypothesis

The central hypothesis is:

> Clinically meaningful information in pathology foundation model embeddings can be selectively suppressed using a low-dimensional latent transformation while preserving most of the original representation geometry and downstream utility.

Under the adversarial interpretation:

> A pathology foundation model can be modified so that it appears functionally intact on most standard evaluations, while a specific target concept is deliberately and selectively destroyed.
