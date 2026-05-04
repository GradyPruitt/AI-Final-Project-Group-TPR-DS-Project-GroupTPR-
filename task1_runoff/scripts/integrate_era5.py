"""
Extract ERA5 hourly time series at each gauge's location and merge into
the existing per-gauge parquet files.

Workflow
--------
1. For each gauge, open the downloaded netCDF.
2. Pick the nearest grid cell to the gauge's lat/lon.
3. Convert variables to friendly units (Kelvin → °C, m → mm, etc.) and
   compute wind speed.
4. Resample any non-hourly fields to hourly (ERA5-Land variables can have
   weird timestamps; ERA5 single-levels is already hourly).
5. Left-join onto the existing gauge parquet (data/processed/gauge_<id>.parquet).
6. Write back the same path. After this step, downstream code that reads
   the parquet will see the new ERA5 columns automatically.

The extra columns added to each gauge parquet:
    era5_t2m_c       2-metre temperature in °C
    era5_tp_mm       total precipitation in mm/hr
    era5_sp_pa       surface pressure in Pa
    era5_wind_speed  10m wind speed in m/s
    era5_wind_dir    wind direction in degrees (0-360, where 0 = north)

These are then ready to be added to default_feature_cols() in dataset.py.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


GAUGES = {
    "20380357": {"lat":  32.7608, "lon": -114.4197},
    "21609641": {"lat":  37.7163, "lon": -119.6657},
}


def _open_era5(nc_path: Path) -> xr.Dataset:
    """Open a netCDF file. Sometimes ERA5 returns instantaneous + accumulated
    variables in separate files; this assumes a single file with everything."""
    return xr.open_dataset(nc_path)


def _normalize_lon(lon: float, ds: xr.Dataset) -> float:
    """ERA5 may use 0–360 longitude convention. Match the dataset's convention."""
    ds_lons = ds["longitude"].values
    if ds_lons.max() > 180 and lon < 0:
        return 360.0 + lon
    return lon


def _to_dataframe(ds: xr.Dataset, lat: float, lon: float) -> pd.DataFrame:
    """Pick nearest grid cell, return tidy hourly dataframe in friendly units."""
    lon_ds = _normalize_lon(lon, ds)
    point = ds.sel(latitude=lat, longitude=lon_ds, method="nearest")

    # Time dimension: ERA5 uses 'valid_time' or 'time' depending on dataset.
    time_var = "valid_time" if "valid_time" in point.coords else "time"

    df = point.to_dataframe().reset_index()
    df = df.rename(columns={time_var: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Convert units. Variable name to expected source name:
    rename = {}
    if "t2m" in df.columns:
        df["era5_t2m_c"] = df["t2m"] - 273.15
    if "tp" in df.columns:
        df["era5_tp_mm"] = df["tp"] * 1000.0     # m → mm
    if "sp" in df.columns:
        df["era5_sp_pa"] = df["sp"]
    if "u10" in df.columns and "v10" in df.columns:
        df["era5_wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
        # Meteorological wind direction: 0° = wind from north
        df["era5_wind_dir"] = (np.degrees(np.arctan2(-df["u10"], -df["v10"])) % 360)

    keep = ["timestamp"] + [c for c in df.columns if c.startswith("era5_")]
    df = df[keep].set_index("timestamp").sort_index()

    # ERA5 single-levels is hourly. If duplicate timestamps appear, keep mean.
    df = df.groupby(df.index).mean()
    return df


def merge_into_parquet(reach_id: str, era5_nc: Path, parquet_path: Path) -> int:
    """Read existing gauge parquet, left-join ERA5 by hourly timestamp, write back.

    Returns number of rows after merge.
    """
    info = GAUGES[reach_id]
    ds = _open_era5(era5_nc)
    era5 = _to_dataframe(ds, lat=info["lat"], lon=info["lon"])

    gauge = pd.read_parquet(parquet_path)
    gauge.index = pd.to_datetime(gauge.index, utc=True)

    merged = gauge.join(era5, how="left")

    # Report missing-data rate for ERA5 columns
    missing = merged[[c for c in merged.columns if c.startswith("era5_")]].isna().mean()
    print(f"  After merge: {len(merged):,} rows.")
    print(f"  ERA5 missing rate per column:")
    for c, frac in missing.items():
        print(f"    {c:20s}  {frac*100:.1f}%")

    # Forward-fill small ERA5 gaps (up to 6h)
    era5_cols = [c for c in merged.columns if c.startswith("era5_")]
    merged[era5_cols] = merged[era5_cols].ffill(limit=6).bfill(limit=6)

    # Drop any rows that still have NaN in ERA5 (shouldn't be any after ffill/bfill
    # if ERA5 covers the period, but be safe).
    n_before = len(merged)
    merged = merged.dropna(subset=era5_cols)
    if len(merged) < n_before:
        print(f"  Dropped {n_before - len(merged):,} rows with missing ERA5 after fill.")

    merged.to_parquet(parquet_path)
    print(f"  → {parquet_path}")
    return len(merged)


def main():
    raw_dir = Path("data/raw_era5")
    proc_dir = Path("data/processed")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Run scripts/download_era5.py first — {raw_dir} not found.")

    for reach_id in GAUGES:
        nc = raw_dir / f"era5_{reach_id}.nc"
        parquet = proc_dir / f"gauge_{reach_id}.parquet"
        if not nc.exists():
            print(f"  Skipping {reach_id}: {nc} not found.")
            continue
        if not parquet.exists():
            print(f"  Skipping {reach_id}: {parquet} not found "
                  f"(run scripts/preprocess.py first).")
            continue
        print(f"\n=== Merging ERA5 into gauge {reach_id} ===")
        merge_into_parquet(reach_id, nc, parquet)


if __name__ == "__main__":
    main()
