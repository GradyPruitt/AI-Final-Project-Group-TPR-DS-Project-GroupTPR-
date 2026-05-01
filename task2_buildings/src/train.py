"""
Training loop for DoriaNET damage classification.

Same overall structure as task1_runoff: load → train with early stopping →
evaluate on test set → save artifacts.

Outputs (per run, to runs/<model>_<timestamp>/):
    model.pt              best validation-loss weights
    history.csv           per-epoch train/val loss + val accuracy
    test_predictions.csv  one row per test building: true label, predicted, top-1, within-1
    summary.json          config + final test metrics
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
from .dataset import DoriaNetBuildingDataset, class_weights, default_transforms, N_CLASSES
from .models import build_model, count_trainable


@dataclass
class TrainConfig:
    # Data
    manifest_csv: str = "data/processed/buildings_manifest.csv"
    image_size: int = 224
    pad_frac: float = 0.10

    # Model
    model: str = "resnet50"   # resnet50 | efficientnet | vit
    freeze_early: bool = True
    dropout: float = 0.3

    # Optimization
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    patience: int = 5
    use_class_weights: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # I/O
    out_dir: str = "runs"


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _epoch(model, loader, loss_fn, device, optimizer=None) -> tuple[float, float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, n = 0.0, 0
    all_true, all_pred = [], []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if is_train:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(xb)
                loss = loss_fn(logits, yb)
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        all_true.append(yb.cpu().numpy())

    preds = np.concatenate(all_pred)
    trues = np.concatenate(all_true)
    avg_loss = total_loss / max(n, 1)
    acc = float((preds == trues).mean())
    return avg_loss, acc, trues, preds


def run(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir) / f"{cfg.model}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    manifest = pd.read_csv(cfg.manifest_csv)
    print(f"Manifest: {len(manifest):,} buildings, {manifest['frame_name'].nunique()} frames")

    train_ds = DoriaNetBuildingDataset(
        manifest, "train",
        transform=default_transforms(train=True, image_size=cfg.image_size),
        pad_frac=cfg.pad_frac,
    )
    val_ds = DoriaNetBuildingDataset(
        manifest, "val",
        transform=default_transforms(train=False, image_size=cfg.image_size),
        pad_frac=cfg.pad_frac,
    )
    test_ds = DoriaNetBuildingDataset(
        manifest, "test",
        transform=default_transforms(train=False, image_size=cfg.image_size),
        pad_frac=cfg.pad_frac,
    )
    print(f"Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    # --- Model ---
    model = build_model(cfg.model, n_classes=N_CLASSES,
                        freeze_early=cfg.freeze_early, dropout=cfg.dropout)
    model.to(cfg.device)
    trainable, total = count_trainable(model)
    print(f"Model: {cfg.model}  trainable {trainable:,}/{total:,} params ({trainable/total:.0%})")

    # --- Loss ---
    if cfg.use_class_weights:
        w = class_weights(manifest, "train").to(cfg.device)
        loss_fn = nn.CrossEntropyLoss(weight=w)
        print(f"Class weights: {[round(x, 3) for x in w.cpu().tolist()]}")
    else:
        loss_fn = nn.CrossEntropyLoss()

    # --- Optimizer (only update params with requires_grad) ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # --- Train with early stopping on val loss ---
    history = []
    best_val = float("inf")
    best_state = None
    bad = 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, _, _ = _epoch(model, train_loader, loss_fn, cfg.device, optimizer)
        va_loss, va_acc, va_true, va_pred = _epoch(model, val_loader, loss_fn, cfg.device)
        dt = time.time() - t0
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc, "time_s": dt,
        })
        print(f"epoch {epoch:3d}  train loss {tr_loss:.4f} acc {tr_acc:.3f}  "
              f"val loss {va_loss:.4f} acc {va_acc:.3f}  ({dt:.1f}s)")
        if va_loss < best_val - 1e-5:
            best_val = va_loss
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

    # --- Test eval ---
    print("\n=== Test evaluation ===")
    _, _, test_true, test_pred = _epoch(model, test_loader, loss_fn, cfg.device)
    test_metrics = M.summary(test_true, test_pred, n_classes=N_CLASSES)

    print(f"  accuracy:       {test_metrics['accuracy']:.4f}")
    print(f"  within-one acc: {test_metrics['within_one_acc']:.4f}")
    print(f"  macro F1:       {test_metrics['macro_f1']:.4f}")
    print(f"  MAE (levels):   {test_metrics['mae_levels']:.4f}")
    print(f"  per-class acc:  {test_metrics['per_class_acc']}")

    pd.DataFrame({
        "y_true": test_true,
        "y_pred": test_pred,
        "correct":   (test_true == test_pred).astype(int),
        "within_one": (np.abs(test_true - test_pred) <= 1).astype(int),
    }).to_csv(out_dir / "test_predictions.csv", index=False)

    summary = {
        "config": asdict(cfg),
        "n_params_trainable": trainable,
        "n_params_total": total,
        "best_val_loss": best_val,
        "test_metrics": test_metrics,
        "out_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nArtifacts: {out_dir}")
    return summary
