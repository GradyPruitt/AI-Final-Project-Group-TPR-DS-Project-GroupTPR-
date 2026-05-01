#!/usr/bin/env python3
"""
Build the DoriaNET per-building manifest CSV from raw frames + masks + JSONs.

Usage
-----
    python scripts/preprocess.py --raw-dir path/to/DoriaNETproject

Expects raw-dir to contain FRAME/, MASK/, and JSON/ subfolders (the standard
layout from AsULearn).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import build_manifest, describe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True, type=Path,
                    help="DoriaNET project folder (containing FRAME/, MASK/, JSON/)")
    ap.add_argument("--output-csv", default="data/processed/buildings_manifest.csv",
                    type=Path, help="Where to write the manifest CSV")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac",   type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Reading from: {args.raw_dir}")
    df = build_manifest(
        raw_dir=args.raw_dir,
        output_csv=args.output_csv,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    describe(df)
    print(f"\n→ {args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
