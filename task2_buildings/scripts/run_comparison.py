#!/usr/bin/env python3
"""
Run all three CV architectures on DoriaNET and produce a comparison table.

Usage
-----
    python scripts/run_comparison.py --epochs 20

    # Subset of models or fewer epochs (for testing):
    python scripts/run_comparison.py --models resnet50 efficientnet --epochs 5

Outputs
-------
    runs/comparison_<timestamp>/
        comparison_table.csv
        comparison_table.md
        individual_runs/  (one folder per (model) — same artifacts as train_baseline)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.train import TrainConfig, run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+",
                    default=["resnet50", "efficientnet", "vit"],
                    choices=["resnet50", "efficientnet", "vit"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest-csv", default="data/processed/buildings_manifest.csv")
    args = ap.parse_args()

    sweep_dir = Path("runs") / f"comparison_{int(time.time())}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "individual_runs").mkdir(exist_ok=True)

    rows = []
    n = len(args.models)
    t0 = time.time()
    for i, model in enumerate(args.models, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{n}] training {model}")
        print("="*70)
        cfg = TrainConfig(
            manifest_csv=args.manifest_csv,
            model=model,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            out_dir=str(sweep_dir / "individual_runs"),
        )
        # ViT needs smaller batch to fit in laptop memory
        if model == "vit":
            cfg.batch_size = min(cfg.batch_size, 16)
        summary = run(cfg)

        m = summary["test_metrics"]
        row = {
            "model":          model,
            "n_params_total": summary["n_params_total"],
            "n_params_train": summary["n_params_trainable"],
            "best_val_loss":  summary["best_val_loss"],
            "test_acc":       m["accuracy"],
            "test_within_one": m["within_one_acc"],
            "test_macro_f1":  m["macro_f1"],
            "test_mae_levels": m["mae_levels"],
        }
        # Per-class accuracy as separate columns
        for c, v in m["per_class_acc"].items():
            row[f"test_acc_class{c}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(sweep_dir / "comparison_table.csv", index=False)

    # Markdown table
    md_path = sweep_dir / "comparison_table.md"
    with open(md_path, "w") as f:
        f.write("# DoriaNET Damage Classification — Model Comparison\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        f.write(f"Configuration: epochs={args.epochs}, patience={args.patience}, "
                f"lr={args.lr}, batch_size={args.batch_size}, seed={args.seed}\n\n")

        f.write("## Test-set metrics\n\n")
        f.write("| Model | Accuracy ↑ | Within-1 Acc ↑ | Macro F1 ↑ | MAE (levels) ↓ | Trainable Params |\n")
        f.write("|-------|-----------|---------------|-----------|----------------|------------------|\n")
        for _, r in df.iterrows():
            f.write(f"| {r['model']} | {r['test_acc']:.4f} | {r['test_within_one']:.4f} | "
                    f"{r['test_macro_f1']:.4f} | {r['test_mae_levels']:.4f} | {r['n_params_train']:,} |\n")

        f.write("\n## Per-class accuracy (Level 0=no damage … Level 5=destroyed)\n\n")
        f.write("| Model | L0 | L1 | L2 | L3 | L4 | L5 |\n")
        f.write("|-------|----|----|----|----|----|----|\n")
        for _, r in df.iterrows():
            cells = " | ".join(f"{r[f'test_acc_class{c}']:.3f}"
                                if not pd.isna(r[f'test_acc_class{c}']) else "—"
                                for c in range(6))
            f.write(f"| {r['model']} | {cells} |\n")

        # Reference: paper achieves 61% top-1 / 90% within-1
        f.write("\n## Comparison against reference paper\n\n")
        f.write("Cheng et al. (2021) report **61% top-1** and **90% within-±1** on the same dataset.\n")
        for _, r in df.iterrows():
            f.write(f"- **{r['model']}**: {r['test_acc']:.1%} top-1, {r['test_within_one']:.1%} within-1\n")

    dt = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Sweep complete: {len(rows)} runs in {dt/60:.1f} min")
    print(f"  {sweep_dir / 'comparison_table.csv'}")
    print(f"  {md_path}")
    print("="*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
