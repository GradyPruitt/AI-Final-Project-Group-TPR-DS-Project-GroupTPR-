"""Hydrology evaluation metrics.

These are the standard metrics for streamflow forecast evaluation:
  - NSE (Nash-Sutcliffe Efficiency): 1 = perfect, 0 = no better than mean.
  - KGE (Kling-Gupta Efficiency): decomposes error into correlation, bias, variability.
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
  - PBIAS (Percent Bias): systematic over/underprediction, in %.
"""

from __future__ import annotations
import numpy as np


def _align(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def rmse(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def mae(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    return float(np.mean(np.abs(yp - yt)))


def nse(y_true, y_pred) -> float:
    """Nash-Sutcliffe Efficiency. Higher is better, max 1.0."""
    yt, yp = _align(y_true, y_pred)
    denom = np.sum((yt - yt.mean()) ** 2)
    if denom == 0:
        return float("nan")
    return float(1 - np.sum((yt - yp) ** 2) / denom)


def kge(y_true, y_pred) -> dict:
    """Kling-Gupta Efficiency plus its three components (r, alpha, beta).

    r     = Pearson correlation
    alpha = std(pred) / std(obs)      (variability ratio)
    beta  = mean(pred) / mean(obs)    (bias ratio)
    kge   = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    """
    yt, yp = _align(y_true, y_pred)
    if yt.std() == 0 or yt.mean() == 0:
        return {"kge": float("nan"), "r": float("nan"),
                "alpha": float("nan"), "beta": float("nan")}
    r = float(np.corrcoef(yt, yp)[0, 1])
    alpha = float(yp.std() / yt.std())
    beta = float(yp.mean() / yt.mean())
    k = float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return {"kge": k, "r": r, "alpha": alpha, "beta": beta}


def pbias(y_true, y_pred) -> float:
    """Percent bias. 0 = unbiased. Positive = model overestimates."""
    yt, yp = _align(y_true, y_pred)
    if yt.sum() == 0:
        return float("nan")
    return float(100.0 * (yp.sum() - yt.sum()) / yt.sum())


def summary(y_true, y_pred) -> dict:
    """One-shot: return all metrics as a flat dict (easy to log/compare)."""
    k = kge(y_true, y_pred)
    return {
        "nse":   nse(y_true, y_pred),
        "rmse":  rmse(y_true, y_pred),
        "mae":   mae(y_true, y_pred),
        "pbias": pbias(y_true, y_pred),
        **k,  # kge, r, alpha, beta
    }
