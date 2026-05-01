#!/usr/bin/env python3
"""
Train a single CV model on the DoriaNET manifest.

Examples
--------
    python scripts/train_baseline.py --model resnet50 --epochs 20
    python scripts/train_baseline.py --model efficientnet --epochs 20
    python scripts/train_baseline.py --model vit --epochs 20 --batch-size 16

Outputs go to runs/<model>_<timestamp>/ — see src/train.py docstring.
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
    ap.add_argument("--manifest-csv", default="data/processed/buildings_manifest.csv")

    ap.add_argument("--model", choices=["resnet50", "efficientnet", "vit"], default="resnet50")
    ap.add_argument("--no-freeze", action="store_true",
                    help="Disable freezing early backbone layers (full fine-tuning)")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--pad-frac", type=float, default=0.10)

    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--no-class-weights", action="store_true",
                    help="Disable inverse-frequency class weights in the loss")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="runs")

    args = ap.parse_args()

    cfg = TrainConfig(
        manifest_csv=args.manifest_csv,
        image_size=args.image_size,
        pad_frac=args.pad_frac,
        model=args.model,
        freeze_early=not args.no_freeze,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        use_class_weights=not args.no_class_weights,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
