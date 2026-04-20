#!/usr/bin/env python3
"""
Preprocess raw NWM + USGS data for one or both gauges → parquet files.

Usage
-----
    # Process both gauges (default):
    python scripts/preprocess.py --raw-dir path/to/RunoffForcastingProject

    # Process one specific gauge:
    python scripts/preprocess.py --raw-dir path/to/RunoffForcastingProject \
        --gauge 20380357

Expects the raw-dir to look like:
    <raw-dir>/20380357/
        streamflow_20380357_202104.csv
        ...
        09520500_Strt_2021-04-20_EndAt_2023-04-21.csv   (USGS obs)
    <raw-dir>/21609641/
        streamflow_21609641_*.csv
        11266500_Strt_*.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Make `src` importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import build_gauge_dataset, describe  # noqa: E402


def find_usgs_csv(gauge_dir: Path) -> Path:
    """Find the USGS observation CSV in a gauge dir.

    It's the one that doesn't start with 'streamflow_'. There should be
    exactly one.
    """
    candidates = [p for p in gauge_dir.glob("*.csv") if not p.name.startswith("streamflow_")]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one USGS CSV in {gauge_dir}, found {len(candidates)}: "
            f"{[p.name for p in candidates]}"
        )
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True, type=Path,
                    help="Top-level directory containing the per-gauge subfolders.")
    ap.add_argument("--out-dir", default="data/processed", type=Path,
                    help="Where to write the processed parquet files (default: data/processed).")
    ap.add_argument("--gauge", default=None,
                    help="Process only this one gauge subfolder. If omitted, process all.")
    args = ap.parse_args()

    raw_dir: Path = args.raw_dir.expanduser().resolve()
    out_dir: Path = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gauge is not None:
        gauge_dirs = [raw_dir / args.gauge]
    else:
        gauge_dirs = [p for p in raw_dir.iterdir() if p.is_dir()]

    if not gauge_dirs:
        print(f"No gauge directories found under {raw_dir}", file=sys.stderr)
        return 1

    for gdir in gauge_dirs:
        reach_id = gdir.name
        try:
            usgs_csv = find_usgs_csv(gdir)
        except FileNotFoundError as e:
            print(f"Skipping {gdir.name}: {e}", file=sys.stderr)
            continue

        out_path = out_dir / f"gauge_{reach_id}.parquet"
        print(f"\n=== Processing gauge {reach_id} ===")
        print(f"  USGS: {usgs_csv.name}")
        print(f"  NWM : {len(list(gdir.glob('streamflow_*.csv')))} monthly files")
        df = build_gauge_dataset(gdir, usgs_csv, output_path=out_path)
        describe(df)
        print(f"  → {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
