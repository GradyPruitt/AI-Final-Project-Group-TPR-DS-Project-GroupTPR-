#!/usr/bin/env python3
"""
Run the full model comparison experiment.

Trains LSTM, GRU, and Transformer on both gauges at lead times 1, 6, 12, 18
(configurable via flags) and writes a comparison table with all metrics.

Usage
-----
    # Full experiment (24 training runs — takes ~30-60 minutes on CPU):
    python scripts/run_comparison.py

    # Quick smoke test with fewer epochs:
    python scripts/run_comparison.py --epochs 5

    # Limit which leads / gauges / models you sweep over:
    python scripts/run_comparison.py --leads 6 18 --models lstm gru

Outputs
-------
    runs/comparison_<timestamp>/
        comparison_table.csv    one row per (gauge, lead, model) run
        comparison_table.md     human-readable markdown table
        individual_runs/        per-run artifacts (same as train_baseline.py)
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
    ap.add_argument("--gauges", nargs="+",
                    default=["20380357", "21609641"],
                    help="Gauge reach IDs to run on.")
    ap.add_argument("--leads", type=int, nargs="+",
                    default=[1, 6, 12, 18],
                    help="Target lead hours to run.")
    ap.add_argument("--models", nargs="+",
                    default=["lstm", "gru", "transformer"],
                    choices=["lstm", "gru", "transformer"],
                    help="Which architectures to run.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--lookback", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", type=str, default="data/processed",
                    help="Where to find gauge_<reach>.parquet files.")
    args = ap.parse_args()

    # One top-level dir for the whole sweep
    sweep_dir = Path("runs") / f"comparison_{int(time.time())}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "individual_runs").mkdir(exist_ok=True)

    rows = []
    n_runs = len(args.gauges) * len(args.leads) * len(args.models)
    run_i = 0
    t0 = time.time()

    for gauge in args.gauges:
        parquet = Path(args.data_dir) / f"gauge_{gauge}.parquet"
        if not parquet.exists():
            print(f"  skipping gauge {gauge}: {parquet} not found", file=sys.stderr)
            continue

        for lead in args.leads:
            for model in args.models:
                run_i += 1
                print(f"\n{'='*70}")
                print(f"[{run_i}/{n_runs}] gauge {gauge} | lead {lead}h | {model}")
                print("="*70)

                cfg = TrainConfig(
                    parquet=str(parquet),
                    target_lead=lead,
                    lookback=args.lookback,
                    model=model,
                    hidden_size=args.hidden_size,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    patience=args.patience,
                    seed=args.seed,
                    out_dir=str(sweep_dir / "individual_runs"),
                )
                summary = run(cfg)

                # Flatten into a single row for the comparison table
                row = {
                    "gauge": gauge,
                    "lead_h": lead,
                    "model": model,
                    "n_params": summary["n_params"],
                }
                for prefix, key in [("nwm_", "metrics_nwm_baseline"),
                                    ("dl_",  "metrics_dl_corrected")]:
                    for metric, val in summary[key].items():
                        row[f"{prefix}{metric}"] = val
                # And the key differences
                row["delta_nse"] = summary["improvement_vs_nwm"]["nse"]
                row["delta_kge"] = summary["improvement_vs_nwm"]["kge"]
                row["delta_rmse"] = row["dl_rmse"] - row["nwm_rmse"]
                rows.append(row)

    # --- Write comparison table ---
    if not rows:
        print("No runs completed.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    csv_path = sweep_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False)

    # --- Human-readable markdown version ---
    md_path = sweep_dir / "comparison_table.md"
    with open(md_path, "w") as f:
        f.write("# Model Comparison Results\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        f.write(f"Configuration: hidden_size={args.hidden_size}, lookback={args.lookback}, "
                f"epochs={args.epochs}, patience={args.patience}, seed={args.seed}\n\n")

        for gauge in args.gauges:
            sub = df[df.gauge == gauge].copy()
            if sub.empty:
                continue
            f.write(f"\n## Gauge {gauge}\n\n")
            # One table per gauge — NSE / KGE / RMSE / MAE, raw NWM vs each DL model
            f.write("### Test-set metrics (DL-corrected forecast vs raw NWM)\n\n")
            f.write("| Lead | Model | NSE ↑ | KGE ↑ | RMSE ↓ | MAE ↓ | PBIAS → 0 | Params |\n")
            f.write("|------|-------|-------|-------|--------|-------|-----------|--------|\n")
            for lead in sorted(sub.lead_h.unique()):
                leadsub = sub[sub.lead_h == lead]
                # First row per lead: raw NWM baseline
                r0 = leadsub.iloc[0]
                f.write(f"| {lead}h | NWM (raw) | {r0['nwm_nse']:.4f} | "
                        f"{r0['nwm_kge']:.4f} | {r0['nwm_rmse']:.3f} | "
                        f"{r0['nwm_mae']:.3f} | {r0['nwm_pbias']:+.2f} | — |\n")
                for _, r in leadsub.iterrows():
                    f.write(f"| {lead}h | {r['model']} | {r['dl_nse']:.4f} | "
                            f"{r['dl_kge']:.4f} | {r['dl_rmse']:.3f} | "
                            f"{r['dl_mae']:.3f} | {r['dl_pbias']:+.2f} | "
                            f"{r['n_params']:,} |\n")
                f.write("\n")

            # Summary: which model won per lead?
            f.write("### Best DL model per lead time (by NSE)\n\n")
            f.write("| Lead | Winner | NSE | Δ vs NWM |\n")
            f.write("|------|--------|-----|----------|\n")
            for lead in sorted(sub.lead_h.unique()):
                leadsub = sub[sub.lead_h == lead]
                best = leadsub.loc[leadsub.dl_nse.idxmax()]
                f.write(f"| {lead}h | **{best['model']}** | "
                        f"{best['dl_nse']:.4f} | {best['delta_nse']:+.4f} |\n")
            f.write("\n")

    dt = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Sweep complete: {len(rows)} runs in {dt/60:.1f} min")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    print("="*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())