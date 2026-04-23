"""
Preprocessing for NWM + USGS runoff data.

What this does
--------------
1. Reads all monthly NWM short-range forecast CSVs for a gauge.
2. Reads the matching USGS 15-minute observation CSV.
3. Aggregates USGS to hourly means (to match NWM's hourly cadence).
4. Pivots NWM so each forecast lead time (1h..18h) becomes a column.
5. Aligns everything on a common hourly UTC timeline.
6. Returns/saves a tidy per-gauge parquet file.

The output parquet has columns:
    timestamp (UTC, hourly)
    usgs_flow             <- ground truth (hourly mean of 15-min)
    nwm_lead_1 .. nwm_lead_18   <- NWM forecast valid at `timestamp` made at init=timestamp - lead hours
    <space for era5_* features>

Notes on NWM file layout
------------------------
Each monthly file `streamflow_<reachID>_YYYYMM.csv` contains every hourly
initialization made during month YYYYMM, with 18 rows per initialization
(one per lead hour). A forecast initialized at 2021-04-30 23:00 is valid
at 2021-05-01 00:00..17:00, so the file named "202104" actually contains
valid times extending into early May. That's fine — we concatenate every
month and align by `valid_time`.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# NWM
# ---------------------------------------------------------------------------

def load_nwm_gauge(gauge_dir: Path) -> pd.DataFrame:
    """Load and concatenate all monthly NWM CSVs in a gauge directory.

    Returns a long dataframe with columns:
        valid_time, init_time, lead_h, nwm_flow, reach_id
    """
    gauge_dir = Path(gauge_dir)
    monthly_files = sorted(gauge_dir.glob("streamflow_*.csv"))
    if not monthly_files:
        raise FileNotFoundError(f"No streamflow_*.csv files under {gauge_dir}")

    frames = []
    for f in monthly_files:
        df = pd.read_csv(f)
        frames.append(df)
    nwm = pd.concat(frames, ignore_index=True)

    # NWM timestamps are stored as YYYY-MM-DD_HH:MM:SS (no timezone).
    # They represent UTC per NOAA convention — we set tz explicitly.
    nwm["valid_time"] = pd.to_datetime(
        nwm["model_output_valid_time"], format="%Y-%m-%d_%H:%M:%S", utc=True
    )
    nwm["init_time"] = pd.to_datetime(
        nwm["model_initialization_time"], format="%Y-%m-%d_%H:%M:%S", utc=True
    )
    nwm["lead_h"] = ((nwm["valid_time"] - nwm["init_time"]).dt.total_seconds() / 3600).astype(int)

    out = nwm.rename(columns={"streamflow_value": "nwm_flow", "streamID": "reach_id"})[
        ["valid_time", "init_time", "lead_h", "nwm_flow", "reach_id"]
    ]

    # Sometimes a given (valid_time, lead_h) pair appears more than once across
    # overlapping monthly files. Keep the first.
    out = out.drop_duplicates(subset=["valid_time", "lead_h"]).reset_index(drop=True)
    return out


def pivot_nwm_by_lead(nwm_long: pd.DataFrame) -> pd.DataFrame:
    """Turn long NWM (one row per lead) into wide (one column per lead)."""
    wide = nwm_long.pivot_table(
        index="valid_time", columns="lead_h", values="nwm_flow", aggfunc="first"
    )
    wide.columns = [f"nwm_lead_{int(h)}" for h in wide.columns]
    wide = wide.sort_index()
    return wide


# ---------------------------------------------------------------------------
# USGS
# ---------------------------------------------------------------------------

def load_usgs_gauge(usgs_csv_path: Path) -> pd.DataFrame:
    """Load USGS 15-minute observations and aggregate to hourly UTC means.

    The two gauges in the midterm dataset use different column names for the
    quality flag ('USGS_GageID' vs '00060_cd'), so we detect it dynamically.

    Returns dataframe indexed by `timestamp` with columns:
        usgs_flow, usgs_estimated
    """
    df = pd.read_csv(usgs_csv_path)
    df["timestamp"] = pd.to_datetime(df["DateTime"], utc=True)

    # Quality flag column: anything that isn't DateTime or the flow value.
    flag_col_candidates = [c for c in df.columns
                           if c not in ("DateTime", "USGSFlowValue", "timestamp")]
    flag_col = flag_col_candidates[0] if flag_col_candidates else None

    df = df.rename(columns={"USGSFlowValue": "usgs_flow"})
    df = df.set_index("timestamp").sort_index()

    # Aggregate to hourly means. Hour-ending labels match NWM's convention.
    hourly = df["usgs_flow"].resample("1h").mean().to_frame()

    # "Estimated" hour if any 15-min value in that hour is flagged with 'e'.
    if flag_col is not None:
        flagged = df[flag_col].astype(str).str.contains("e", na=False).astype(int)
        qual = flagged.resample("1h").max().astype(bool)
        hourly["usgs_estimated"] = qual.reindex(hourly.index, fill_value=False)
    else:
        hourly["usgs_estimated"] = False

    return hourly


# ---------------------------------------------------------------------------
# Merge & clean
# ---------------------------------------------------------------------------

def build_gauge_dataset(
    gauge_dir: Path,
    usgs_csv: Path,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """End-to-end: build one clean per-gauge hourly dataset.

    Parameters
    ----------
    gauge_dir : directory containing streamflow_<reach>_YYYYMM.csv files
                 AND the usgs CSV (typical layout from the midterm).
    usgs_csv  : explicit path to the USGS CSV inside that directory.
    output_path : if given, write parquet here.

    Returns
    -------
    Hourly dataframe indexed by UTC timestamp with columns:
        usgs_flow, usgs_estimated, nwm_lead_1..nwm_lead_18
    """
    nwm_long = load_nwm_gauge(gauge_dir)
    nwm_wide = pivot_nwm_by_lead(nwm_long)
    nwm_wide.index.name = "timestamp"

    usgs_hourly = load_usgs_gauge(usgs_csv)
    usgs_hourly.index.name = "timestamp"

    # Inner-join on timestamp so every row has both observed and forecast.
    merged = usgs_hourly.join(nwm_wide, how="inner").sort_index()


    era5 = pd.read_csv("era5_point_timeseries.csv", parse_dates=["time"])
    era5 = era5.set_index("time").sort_index()

    # align ERA5 to hourly timeline
    era5 = era5.resample("1h").ffill()

    # merge into main dataframe
    merged = merged.join(era5, how="left")

    # forward fill remaining small gaps
    merged = merged.ffill()

    print(merged.columns)
    
    # Basic cleaning
    lead_cols = [c for c in merged.columns if c.startswith("nwm_lead_")]

    # 1. Drop rows where *all* NWM leads are NaN (shouldn't happen, but safe).
    merged = merged.dropna(subset=lead_cols, how="all")

    # 2. Forward-fill short USGS gaps up to 3 hours (sensor blips).
    merged["usgs_flow"] = merged["usgs_flow"].ffill(limit=3)

    # 3. Drop rows that still have NaN in usgs_flow (longer gaps).
    merged = merged.dropna(subset=["usgs_flow"])

    # 4. Any remaining NWM lead NaNs: linear-interpolate small gaps.
    merged[lead_cols] = merged[lead_cols].interpolate(method="time", limit=6)
    merged = merged.dropna(subset=lead_cols)

    # 5. Clip negative flows to 0 (nonphysical).
    merged["usgs_flow"] = merged["usgs_flow"].clip(lower=0)
    for c in lead_cols:
        merged[c] = merged[c].clip(lower=0)

    # 6. Convenience: residuals at each lead (obs - forecast). These are what
    #    we actually want the DL model to learn per the Frame et al. paper.
    for h in range(1, 19):
        col = f"nwm_lead_{h}"
        if col in merged.columns:
            merged[f"resid_lead_{h}"] = merged["usgs_flow"] - merged[col]

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(output_path)

    return merged


# ---------------------------------------------------------------------------
# Train / val / test split (per midterm spec)
# ---------------------------------------------------------------------------

TRAIN_END   = "2022-06-30 23:00:00+00:00"
VAL_END     = "2022-09-30 23:00:00+00:00"
TEST_START  = "2022-10-01 00:00:00+00:00"


def split_train_val_test(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Chronological split. Test window is hard-coded from the assignment.

    Train: Apr 2021 - Jun 2022
    Val:   Jul 2022 - Sep 2022
    Test:  Oct 2022 - Apr 2023   (NEVER use for training/tuning)
    """
    train = df.loc[:TRAIN_END]
    val = df.loc[TRAIN_END:VAL_END].iloc[1:]  # exclusive of TRAIN_END
    test = df.loc[TEST_START:]
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def describe(df: pd.DataFrame) -> None:
    """Print a quick summary — handy for sanity-checking preprocessing."""
    print(f"rows: {len(df):,}")
    print(f"time range: {df.index.min()}  →  {df.index.max()}")
    print(f"usgs_flow  min/mean/max: "
          f"{df.usgs_flow.min():.3f} / {df.usgs_flow.mean():.3f} / {df.usgs_flow.max():.3f}")
    print(f"nwm_lead_1 min/mean/max: "
          f"{df.nwm_lead_1.min():.3f} / {df.nwm_lead_1.mean():.3f} / {df.nwm_lead_1.max():.3f}")
    print(f"residual lead-1 mean/std: "
          f"{df.resid_lead_1.mean():.3f} / {df.resid_lead_1.std():.3f}")
    print(f"missing values per column:\n{df.isna().sum()[df.isna().sum() > 0]}")
