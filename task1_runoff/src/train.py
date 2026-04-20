"""
Training loop for the runoff forecasting models.

Intentionally framework-light (no Lightning, no Hydra) so it's easy to read
and modify for the report. Everything important is in one screenful.

What it does
------------
1. Loads a preprocessed per-gauge parquet.
2. Splits chronologically into train/val/test (Oct 2022 onward is test).
3. Builds PyTorch datasets with shared scalers (no leakage).
4. Trains the chosen model with early stopping on val loss.
5. Evaluates on the test set — both the raw NWM forecast and the
   DL-corrected forecast — and prints a side-by-side metrics comparison.

Usage from command line (see scripts/train_baseline.py)
--------------------------------------------------------
    python scripts/train_baseline.py \
        --parquet data/processed/gauge_20380357.parquet \
        --model lstm --target-lead 6 --epochs 30
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import metrics as M
from .dataset import RunoffWindowDataset, default_feature_cols
from .models import build_model
from .preprocessing import split_train_val_test


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    parquet: str = "data/processed/gauge_20380357.parquet"
    target_lead: int = 6             # hours
    lookback: int = 48               # hours of history
    feature_cols: list = field(default_factory=default_feature_cols)
    target_kind: str = "residual"    # or 'flow'

    # Model
    model: str = "lstm"              # 'lstm' | 'gru' | 'transformer'
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2

    # Optimization
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 30
    patience: int = 5                # early stopping patience on val loss
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # I/O
    out_dir: str = "runs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _epoch_loss(model, loader, loss_fn, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if train_mode:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                pred = model(xb)
                loss = loss_fn(pred, yb)
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def _predict(model, loader, device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir) / f"{cfg.model}_lead{cfg.target_lead}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---------------------------------------------------------
    df = pd.read_parquet(cfg.parquet)
    splits = split_train_val_test(df)
    print(f"Splits: train={len(splits['train'])}, "
          f"val={len(splits['val'])}, test={len(splits['test'])}")

    train_ds = RunoffWindowDataset(
        splits["train"], cfg.feature_cols,
        target_lead=cfg.target_lead, lookback=cfg.lookback,
        target_kind=cfg.target_kind,
    )
    scalers = (train_ds.feature_scaler, train_ds.target_scaler)
    val_ds = RunoffWindowDataset(
        splits["val"], cfg.feature_cols,
        target_lead=cfg.target_lead, lookback=cfg.lookback,
        target_kind=cfg.target_kind, scalers=scalers,
    )
    test_ds = RunoffWindowDataset(
        splits["test"], cfg.feature_cols,
        target_lead=cfg.target_lead, lookback=cfg.lookback,
        target_kind=cfg.target_kind, scalers=scalers,
    )
    print(f"Windowed: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False)

    # --- Model --------------------------------------------------------
    model_kwargs = dict(hidden_size=cfg.hidden_size,
                        num_layers=cfg.num_layers,
                        dropout=cfg.dropout)
    if cfg.model == "transformer":
        model_kwargs["lookback"] = cfg.lookback
    model = build_model(cfg.model, n_features=len(cfg.feature_cols), **model_kwargs)
    model.to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg.model}  params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    # --- Train --------------------------------------------------------
    best_val = float("inf")
    best_state = None
    history = []
    bad = 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = _epoch_loss(model, train_loader, loss_fn, cfg.device, optimizer)
        va = _epoch_loss(model, val_loader,   loss_fn, cfg.device)
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va, "time_s": dt})
        print(f"epoch {epoch:3d}  train {tr:.4f}  val {va:.4f}  ({dt:.1f}s)")
        if va < best_val - 1e-5:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "model.pt")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # --- Evaluate on test --------------------------------------------
    print("\n=== Test evaluation ===")

    pred_scaled = _predict(model, test_loader, cfg.device)
    pred = test_ds.unscale_targets(pred_scaled)    # back to physical units

    usgs_test = test_ds.raw_usgs_obs()
    nwm_test  = test_ds.raw_nwm_forecast()

    if cfg.target_kind == "residual":
        corrected = nwm_test + pred
    else:
        corrected = pred

    metrics_nwm = M.summary(usgs_test, nwm_test)
    metrics_dl  = M.summary(usgs_test, corrected)

    improvement = {k: metrics_dl[k] - metrics_nwm[k]
                   for k in ["nse", "kge", "r"]}
    print("\n           NWM baseline    DL corrected    Δ")
    for k in ["nse", "kge", "rmse", "mae", "pbias", "r"]:
        print(f"  {k:6s}  {metrics_nwm[k]:>12.4f}   {metrics_dl[k]:>12.4f}   "
              f"{metrics_dl[k] - metrics_nwm[k]:+.4f}")

    # Save per-timestep predictions for report plots
    results = pd.DataFrame({
        "timestamp": test_ds.forecast_timestamps(),
        "usgs_obs": usgs_test,
        "nwm_raw": nwm_test,
        "nwm_corrected": corrected,
        "residual_pred": pred if cfg.target_kind == "residual" else pred - nwm_test,
    })
    results.to_csv(out_dir / "test_predictions.csv", index=False)

    summary_dict = {
        "config": asdict(cfg),
        "n_params": n_params,
        "best_val_loss": best_val,
        "metrics_nwm_baseline": metrics_nwm,
        "metrics_dl_corrected": metrics_dl,
        "improvement_vs_nwm": improvement,
        "out_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2, default=str)

    print(f"\nArtifacts: {out_dir}")
    return summary_dict
