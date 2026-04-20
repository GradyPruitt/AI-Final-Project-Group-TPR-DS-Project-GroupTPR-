"""
PyTorch Dataset for the runoff post-processing task.

Given a preprocessed per-gauge hourly dataframe (from preprocessing.py), this
module builds (X, y) pairs where:

    X : a lookback window of shape (L, F)
        - last L hours of past USGS observations
        - last L hours of past NWM forecasts at each relevant lead (or a subset)
        - optional extra features (e.g. ERA5 once you add it)
    y : the residual at the target lead time, valid at the next hour(s).
          residual = USGS(t+h) - NWM_forecast_for_t+h_made_at_t

We intentionally predict the RESIDUAL rather than the flow directly, following
Han & Morrison (2022). At inference time the corrected forecast is simply
    corrected(t+h) = NWM(t+h) + residual_hat(t+h)

Why this framing?
-----------------
- It leverages NWM's physical skill as a strong baseline, so the DL model only
  has to learn the correction term (a smaller-magnitude signal).
- It's the exact setup from the paper Prof handed out.
- It keeps the targets at a similar scale across gauges, which helps the
  network train stably.

If you want to try "predict the flow directly" as an alternative, flip the
`target_kind` arg to "flow".
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Feature scaling
# ---------------------------------------------------------------------------

class StandardScaler1D:
    """Minimal fit/transform scaler that handles torch/numpy and stores stats."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, arr: np.ndarray):
        arr = np.asarray(arr, dtype=np.float32)
        self.mean_ = arr.mean(axis=0, keepdims=True)
        self.std_ = arr.std(axis=0, keepdims=True) + 1e-6
        return self

    def transform(self, arr):
        arr = np.asarray(arr, dtype=np.float32)
        return (arr - self.mean_) / self.std_

    def inverse_transform(self, arr):
        arr = np.asarray(arr, dtype=np.float32)
        return arr * self.std_ + self.mean_

    def fit_transform(self, arr):
        return self.fit(arr).transform(arr)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RunoffWindowDataset(Dataset):
    """
    Parameters
    ----------
    df : dataframe from preprocessing.build_gauge_dataset(...), indexed hourly.
    feature_cols : which columns to stack into the input window. Typically
        ['usgs_flow'] + [f'nwm_lead_{h}' for h in range(1, 19)]
    target_lead : forecast lead time in hours to predict (1..18).
    lookback : how many past hours of context to feed the network.
    target_kind : 'residual' (default) or 'flow'.
        - 'residual': y = USGS(t+target_lead) - NWM_lead_{target_lead}(t+target_lead)
          (i.e. the thing valid at t+target_lead, whose forecast was made at time t)
        - 'flow'    : y = USGS(t+target_lead)
    scalers : optional pre-fit (feature_scaler, target_scaler) pair. Pass None
        when fitting on training data; pass the fit scalers from the training
        dataset when building val/test so there's no leakage.

    Returns items of shape (X, y) where
        X.shape == (lookback, len(feature_cols))
        y is a scalar
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_lead: int,
        lookback: int = 48,
        target_kind: str = "residual",
        scalers: tuple[StandardScaler1D, StandardScaler1D] | None = None,
    ):
        assert target_kind in ("residual", "flow"), target_kind
        assert 1 <= target_lead <= 18

        self.feature_cols = feature_cols
        self.target_lead = int(target_lead)
        self.lookback = int(lookback)
        self.target_kind = target_kind

        # Pull the feature matrix and target series
        F = df[feature_cols].to_numpy(dtype=np.float32)

        if target_kind == "residual":
            y_col = f"resid_lead_{self.target_lead}"
        else:
            y_col = "usgs_flow"
        # For target alignment: the forecast made at index t for lead h is valid
        # at index t+h. So y_t = df[y_col].iloc[t + target_lead] if residual is
        # already stored aligned to valid_time, OR just take resid at t+h since
        # our preprocessing stored resid as obs(valid_time) - nwm(valid_time at
        # lead h) indexed by valid_time.
        #
        # Convention used here: the window ends at valid_time = t_end. The
        # target is the value valid at t_end + target_lead. This means we
        # simulate "standing at t_end, predicting h hours into the future".
        y_full = df[y_col].to_numpy(dtype=np.float32)

        # Keep the raw NWM forecast at the target valid time so we can report
        # the corrected forecast without re-joining later.
        self.nwm_at_target = df[f"nwm_lead_{self.target_lead}"].to_numpy(dtype=np.float32)
        self.usgs_at_target = df["usgs_flow"].to_numpy(dtype=np.float32)
        self.index = df.index

        n = len(df)
        # Valid start indices: need `lookback` history and `target_lead` future.
        starts = np.arange(n - lookback - self.target_lead + 1)

        # Build windows eagerly (small-ish dataset, this is fine in RAM).
        self._X_raw = np.stack([F[s: s + lookback] for s in starts], axis=0)  # (N, L, F)
        self._y_raw = y_full[starts + lookback + self.target_lead - 1]        # (N,)
        self._target_idx = starts + lookback + self.target_lead - 1           # for later

        # Fit or apply scalers
        if scalers is None:
            self.feature_scaler = StandardScaler1D().fit(self._X_raw.reshape(-1, F.shape[1]))
            self.target_scaler = StandardScaler1D().fit(self._y_raw.reshape(-1, 1))
        else:
            self.feature_scaler, self.target_scaler = scalers

        flat = self.feature_scaler.transform(self._X_raw.reshape(-1, F.shape[1]))
        self.X = flat.reshape(self._X_raw.shape).astype(np.float32)
        self.y = self.target_scaler.transform(self._y_raw.reshape(-1, 1)).ravel().astype(np.float32)

    # --- pytorch API ---
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

    # --- helpers for evaluation ---
    def unscale_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    def forecast_timestamps(self) -> pd.DatetimeIndex:
        """The valid times of the targets, in order."""
        return self.index[self._target_idx]

    def raw_nwm_forecast(self) -> np.ndarray:
        """The uncorrected NWM forecast for each target timestamp."""
        return self.nwm_at_target[self._target_idx]

    def raw_usgs_obs(self) -> np.ndarray:
        """The observed USGS flow at each target timestamp."""
        return self.usgs_at_target[self._target_idx]


# ---------------------------------------------------------------------------
# Convenience: default feature set
# ---------------------------------------------------------------------------

def default_feature_cols() -> list[str]:
    """A reasonable default input set: USGS history + all 18 NWM leads.

    Feel free to trim this (e.g. to just usgs_flow + nwm_lead_1) for speed,
    or expand it once ERA5 is added.
    """
    return ["usgs_flow"] + [f"nwm_lead_{h}" for h in range(1, 19)]
