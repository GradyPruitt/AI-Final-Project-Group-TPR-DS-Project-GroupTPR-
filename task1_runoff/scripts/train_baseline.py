#!/usr/bin/env python3
"""
Train one of the post-processing models on a preprocessed gauge.

Examples
--------
    # Baseline LSTM, 6h lead, default hyperparams:
    python scripts/train_baseline.py \
        --parquet data/processed/gauge_21609641.parquet \
        --model lstm --target-lead 6 --epochs 30

    # GRU variant (after you implement it in src/models.py):
    python scripts/train_baseline.py \
        --parquet data/processed/gauge_21609641.parquet \
        --model gru --target-lead 6 --epochs 30

    # Transformer (after you implement it in src/models.py):
    python scripts/train_baseline.py \
        --parquet data/processed/gauge_21609641.parquet \
        --model transformer --target-lead 6 --epochs 30 \
        --hidden-size 64

Outputs go to `runs/<model>_lead<h>_<timestamp>/`:
    - model.pt                saved weights
    - history.csv             per-epoch train/val loss
    - test_predictions.csv    timestamp, obs, raw NWM, corrected NWM, residual_pred
    - summary.json            metrics comparing NWM baseline vs DL correction
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import TrainConfig, run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data
    ap.add_argument("--parquet", required=True, type=str,
                    help="Path to the preprocessed gauge parquet.")
    ap.add_argument("--target-lead", type=int, default=6,
                    help="Forecast lead time in hours (1..18). Default: 6.")
    ap.add_argument("--lookback", type=int, default=48,
                    help="Hours of history to feed the model. Default: 48.")
    ap.add_argument("--target-kind", choices=["residual", "flow"], default="residual",
                    help="Predict the residual (default) or the flow directly.")

    # Model
    ap.add_argument("--model", choices=["lstm", "gru", "transformer"], default="lstm")
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)

    # Optimization
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    # I/O
    ap.add_argument("--out-dir", type=str, default="runs")

    args = ap.parse_args()

    cfg = TrainConfig(
        parquet=args.parquet,
        target_lead=args.target_lead,
        lookback=args.lookback,
        target_kind=args.target_kind,
        model=args.model,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
