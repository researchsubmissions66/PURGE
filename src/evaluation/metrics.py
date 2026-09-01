"""
Metrics.

Note on chance level: every AUC here is macro one-vs-rest, whose chance value is
0.50 regardless of the number of classes. A successful erasure therefore drives
the target AUC to ~0.50, not to 1/num_classes.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


class MetricError(ValueError):
    pass


def macro_ovr_auc(labels, probs, num_classes, strict=True):
    """
    Macro one-vs-rest ROC AUC that degrades explicitly rather than silently.

    labels: [N] ints
    probs:  [N] P(class 1) when num_classes == 2, else [N, num_classes]
    strict: if True, raise MetricError when the AUC is undefined; if False,
            return (None, reason).

    Returns (auc, note). `note` is '' when every class was scored, otherwise it
    records which classes were skipped.

    Never returns a placeholder value. A silent 0.5 is indistinguishable from a
    genuine chance-level result, which is the exact quantity this project measures.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    if labels.size == 0:
        return _fail("no samples", strict)

    present = np.unique(labels)
    if present.size < 2:
        return _fail(
            f"only class {present.tolist()} present in the evaluation split; "
            "AUC is undefined", strict
        )

    if num_classes == 2:
        if probs.ndim != 1:
            raise MetricError(f"binary AUC expects 1-D probs, got shape {probs.shape}")
        return float(roc_auc_score(labels, probs)), ''

    if probs.ndim != 2 or probs.shape[1] != num_classes:
        raise MetricError(
            f"expected probs of shape [N, {num_classes}], got {probs.shape}"
        )

    aucs, skipped = [], []
    for c in range(num_classes):
        y_bin = (labels == c)
        if y_bin.all() or not y_bin.any():
            skipped.append(c)
            continue
        aucs.append(roc_auc_score(y_bin, probs[:, c]))

    if not aucs:
        return _fail("no class had both positive and negative examples", strict)

    note = ''
    if skipped:
        note = f"macro over {len(aucs)}/{num_classes} classes; absent from split: {skipped}"
    return float(np.mean(aucs)), note


def _fail(reason, strict):
    if strict:
        raise MetricError(reason)
    return None, reason


def selective_degradation_score(target_drop, control_drops, eps=1e-3):
    """
    SDS from plan section 17: target degradation per unit of collateral damage.

        SDS = dP_target / (eps + mean_k |dP_k|)

    High is good: the target collapsed and the controls did not.

    CAUTION: with collateral near zero the denominator is dominated by `eps` and
    SDS explodes, so a method that barely erases but leaves controls untouched can
    outscore a method that actually works. Measured: a relevance-ranked eraser
    scored SDS 46.5 against plain SVD's 38.8 while erasing far less (target 0.726
    vs 0.466). Always report the target drop alongside SDS; never rank methods by
    SDS alone.
    """
    if not control_drops:
        raise ValueError("selective_degradation_score needs at least one control")
    collateral = float(np.mean([abs(d) for d in control_drops]))
    return float(target_drop) / (eps + collateral)
