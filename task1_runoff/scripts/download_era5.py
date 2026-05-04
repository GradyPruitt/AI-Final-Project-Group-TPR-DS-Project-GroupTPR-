"""
Download ERA5 hourly reanalysis data for both gauge basins.

CDS calculates request "cost" based on the unfiltered global grid size, not
what the area filter returns. Even a single-year, all-variables request can
trip their cost limit. To get under the limit, we split into many small
requests: one per (gauge, year-month, variable). That keeps each chunk
small enough to clear the cost check, at the price of submitting many
requests sequentially through CDS's queue.

For our scope (2 gauges × 25 months × 5 variables ≈ 250 requests) this
takes a while in queue time — plan for it to run overnight. Each
individual request is tiny (a few hundred KB) and downloads in seconds
once it's out of the queue.

The downloads are resumable. If the script is interrupted, just rerun it;
it skips any (gauge, year-month, variable) chunk that already has a file.
"""

from __future__ import annotations
from pathlib import Path
import cdsapi


# Gauge metadata — gauge_id → (lat, lon, USGS site number, name)
GAUGES = {
    "20380357": {
        "lat":  32.7608,
        "lon": -114.4197,
        "usgs": "09520500",
        "name": "Gila River near Dome, AZ",
    },
    "21609641": {
        "lat":  37.7163,
        "lon": -119.6657,
        "usgs": "11266500",
        "name": "Merced River basin, CA",
    },
}


# Spatial window around each gauge — small box keeps the file tiny.
# CDS takes [North, West, South, East]. 0.25° on each side ≈ 25 km radius.
HALF_BOX = 0.25


VARIABLES = [
    "2m_temperature",
    "total_precipitation",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]


# Project window: April 2021 through April 2023 inclusive.
YEAR_MONTHS = (
    [(2021, m) for m in range(4, 13)] +     # Apr-Dec 2021
    [(2022, m) for m in range(1, 13)] +     # Jan-Dec 2022
    [(2023, m) for m in range(1, 5)]        # Jan-Apr 2023
)
DAYS  = [f"{d:02d}" for d in range(1, 32)]
HOURS = [f"{h:02d}:00" for h in range(0, 24)]


def _request_chunk(client, area, variable, year, month, out_path):
    """Submit one small (gauge, month, variable) request to CDS."""
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [variable],
            "year":  [str(year)],
            "month": [f"{month:02d}"],
            "day":   DAYS,
            "time":  HOURS,
            "area":  area,
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(out_path),
    )


def download_one(reach_id: str, info: dict, outdir: Path) -> Path:
    """Download per gauge in (year-month, variable) chunks, then concat.

    Final output: one `era5_<reach_id>.nc` per gauge, with all variables
    aligned on the same hourly time index.
    """
    final = outdir / f"era5_{reach_id}.nc"
    if final.exists():
        print(f"  already exists, skipping: {final}")
        return final

    chunk_dir = outdir / f"chunks_{reach_id}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    area = [info["lat"] + HALF_BOX,
            info["lon"] - HALF_BOX,
            info["lat"] - HALF_BOX,
            info["lon"] + HALF_BOX]

    print(f"\n=== Requesting ERA5 for gauge {reach_id} ({info['name']}) ===")
    print(f"  area (N, W, S, E): {area}")
    print(f"  splitting into {len(YEAR_MONTHS)} months × {len(VARIABLES)} variables = "
          f"{len(YEAR_MONTHS) * len(VARIABLES)} small requests")

    client = cdsapi.Client()

    chunk_paths = []
    total = len(YEAR_MONTHS) * len(VARIABLES)
    i = 0
    for (year, month) in YEAR_MONTHS:
        for var in VARIABLES:
            i += 1
            cpath = chunk_dir / f"{var}_{year}{month:02d}.nc"
            chunk_paths.append(cpath)
            if cpath.exists():
                continue
            print(f"  [{i}/{total}] {var} {year}-{month:02d}...", flush=True)
            _request_chunk(client, area, var, year, month, cpath)

    print(f"  all {total} chunks ready, concatenating...")

    # Concatenate: group all chunks by variable first (concat along time),
    # then merge across variables.
    import xarray as xr

    by_var = {}
    for cpath in chunk_paths:
        var = cpath.stem.rsplit("_", 1)[0]
        by_var.setdefault(var, []).append(cpath)

    # Open and concat each variable's monthly chunks along time
    var_datasets = []
    for var, paths in by_var.items():
        paths = sorted(paths)
        dsets = [xr.open_dataset(p) for p in paths]
        time_dim = "valid_time" if "valid_time" in dsets[0].dims else "time"
        combined = xr.concat(dsets, dim=time_dim).sortby(time_dim)
        combined = combined.drop_duplicates(dim=time_dim)
        var_datasets.append(combined)
        for d in dsets:
            d.close()

    # Merge variables (they share the same time/lat/lon coordinates)
    merged = xr.merge(var_datasets, compat="override")
    merged.to_netcdf(final)
    for d in var_datasets:
        d.close()

    print(f"  → {final}  ({final.stat().st_size / 1e6:.1f} MB)")
    return final


def main():
    outdir = Path("data/raw_era5")
    outdir.mkdir(parents=True, exist_ok=True)

    for reach_id, info in GAUGES.items():
        try:
            download_one(reach_id, info, outdir)
        except Exception as e:
            print(f"  FAILED for {reach_id}: {e}")
            print(f"  (You can re-run this script; it will skip already-downloaded chunks.)")
            raise


if __name__ == "__main__":
    main()
