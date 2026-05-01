"""
Evaluation metrics for ordinal damage classification.

In ordinal classification, predicting "damage level 4" when the truth is
"level 3" is a less serious error than predicting "level 0" when the truth
is "level 5". Standard accuracy treats both as equally wrong. We report
metrics that reflect this ordering:

  - accuracy           top-1 accuracy (fraction with exactly correct level)
  - within_one_acc     fraction predicted within ±1 of the true level
                       (this is what Cheng et al. 2021 emphasizes)
  - macro_f1           per-class F1 averaged equally across classes
                       (accounts for class imbalance)
  - mae_levels         mean absolute error in level units
                       (continuous-space view of the ordinal error)
  - confusion_matrix   raw 6x6 matrix
"""

from __future__ import annotations
import numpy as np


def _align(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true).ravel().astype(int)
    yp = np.asarray(y_pred).ravel().astype(int)
    assert yt.shape == yp.shape, f"shape mismatch {yt.shape} vs {yp.shape}"
    return yt, yp


def accuracy(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    return float((yt == yp).mean())


def within_one_acc(y_true, y_pred) -> float:
    """Fraction of predictions within ±1 level of the truth.

    This is the metric Cheng et al. (2021) reports as their main result
    (61% top-1, 90% within ±1). Their reasoning: in disaster response, a
    level-3 building flagged as level-4 is operationally similar.
    """
    yt, yp = _align(y_true, y_pred)
    return float((np.abs(yt - yp) <= 1).mean())


def mae_levels(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    return float(np.abs(yt.astype(float) - yp.astype(float)).mean())


def confusion_matrix(y_true, y_pred, n_classes: int = 6) -> np.ndarray:
    yt, yp = _align(y_true, y_pred)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(yt, yp):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def macro_f1(y_true, y_pred, n_classes: int = 6) -> float:
    """Macro-averaged F1 score (averages F1 across classes equally)."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    f1s = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1s))


def per_class_accuracy(y_true, y_pred, n_classes: int = 6) -> dict[int, float]:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    out = {}
    for c in range(n_classes):
        n = cm[c, :].sum()
        out[c] = float(cm[c, c] / n) if n > 0 else float("nan")
    return out


def summary(y_true, y_pred, n_classes: int = 6) -> dict:
    return {
        "accuracy":        accuracy(y_true, y_pred),
        "within_one_acc":  within_one_acc(y_true, y_pred),
        "macro_f1":        macro_f1(y_true, y_pred, n_classes),
        "mae_levels":      mae_levels(y_true, y_pred),
        "per_class_acc":   per_class_accuracy(y_true, y_pred, n_classes),
    }
