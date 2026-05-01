"""
Preprocessing for the DoriaNET hurricane damage dataset.

What this does
--------------
1. Reads all JSON annotations in the DoriaNET project folder.
2. Extracts per-building records: which frame, which mask, damage level.
3. Verifies that the frame and mask files exist on disk.
4. Computes the bounding box of each building from its binary mask.
5. Builds train/val/test splits at the FRAME level (not building level) so
   that no frame appears in two splits — this prevents the model from
   "memorizing" a frame's neighbors.
6. Writes a single tidy manifest CSV with one row per building.

Output schema (data/processed/buildings_manifest.csv)
-----------------------------------------------------
    building_id           e.g. "B004"
    frame_name            e.g. "1_0373" (just the stem, no extension)
    frame_path            absolute path to the frame .jpg
    mask_name             e.g. "1_0373_B0XX_0_Level1.jpg"
    mask_path             absolute path to the mask .jpg
    damage_level          int 0-5 (FEMA states 0-4 + Cheng et al.'s level 5)
    bbox_x0, bbox_y0      top-left of mask bounding box (pixels)
    bbox_x1, bbox_y1      bottom-right of mask bounding box (pixels)
    bbox_w, bbox_h        width and height in pixels
    dataset_group         "1", "2", or "3" — the original video the frame came from
    split                 "train", "val", or "test"

Why frame-level splitting matters
---------------------------------
Buildings within the same frame are visually correlated (same lighting,
same vegetation, often adjacent in space). If we shuffled buildings into
train/test independently, the model would see "neighboring" buildings of
the same frame in both splits and we'd overestimate generalization.
Splitting at the frame level is the standard fix.
"""

from __future__ import annotations
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _load_json_safe(path: Path) -> dict:
    """DoriaNET JSONs include literal `NaN` which isn't valid JSON. Patch and parse."""
    txt = path.read_text()
    # Replace bare NaN tokens (with surrounding whitespace) with null
    txt = re.sub(r"\bNaN\b", "null", txt)
    return json.loads(txt)


def _parse_buildings(json_data: dict) -> list[dict]:
    """Each entry in json['Buildings'] looks like:
        [bldg_id, "lat, lon", mask_filename, level, x, y, z]
    """
    out = []
    for b in json_data.get("Buildings", []):
        if len(b) < 4:
            continue
        out.append({
            "building_id": b[0],
            "lat_lon": b[1] if len(b) > 1 else None,
            "mask_name": b[2] if len(b) > 2 else None,
            "damage_level": int(b[3]),
        })
    return out


# ---------------------------------------------------------------------------
# Bounding box from mask
# ---------------------------------------------------------------------------

def _mask_bbox(mask_path: Path, threshold: int = 128) -> tuple[int, int, int, int] | None:
    """Compute the bounding box of the white region in a binary mask.

    Returns (x0, y0, x1, y1) in pixel coordinates, or None if the mask is empty.
    """
    arr = np.asarray(Image.open(mask_path).convert("L"))
    ys, xs = np.where(arr > threshold)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# ---------------------------------------------------------------------------
# Train/val/test split at the frame level
# ---------------------------------------------------------------------------

def _frame_level_split(
    frame_names: list[str],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    """Assign each frame to train/val/test deterministically.

    Returns dict mapping frame_name -> "train"/"val"/"test".
    """
    rng = np.random.default_rng(seed)
    frames = sorted(set(frame_names))   # sort for reproducibility
    rng.shuffle(frames)

    n = len(frames)
    n_train = int(round(train_frac * n))
    n_val   = int(round(val_frac   * n))
    splits = {}
    for i, fr in enumerate(frames):
        if i < n_train:
            splits[fr] = "train"
        elif i < n_train + n_val:
            splits[fr] = "val"
        else:
            splits[fr] = "test"
    return splits


# ---------------------------------------------------------------------------
# End-to-end builder
# ---------------------------------------------------------------------------

def build_manifest(
    raw_dir: Path,
    output_csv: Path | None = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the per-building manifest.

    Parameters
    ----------
    raw_dir : Path to the DoriaNET project folder containing FRAME/, MASK/, JSON/
    output_csv : if given, write CSV here.

    Returns
    -------
    DataFrame with one row per building.
    """
    raw_dir = Path(raw_dir)
    frame_dir = raw_dir / "FRAME"
    mask_dir = raw_dir / "MASK"
    json_dir = raw_dir / "JSON"

    for d in (frame_dir, mask_dir, json_dir):
        if not d.exists():
            raise FileNotFoundError(f"Missing expected DoriaNET subfolder: {d}")

    rows = []
    n_skipped_missing_files = 0
    n_skipped_empty_mask = 0

    for jpath in sorted(json_dir.glob("*.json")):
        try:
            data = _load_json_safe(jpath)
        except json.JSONDecodeError as e:
            print(f"  WARNING: could not parse {jpath.name}: {e}")
            continue

        frame_stem = jpath.stem  # e.g. "1_0373"
        frame_path = frame_dir / f"{frame_stem}.jpg"
        if not frame_path.exists():
            n_skipped_missing_files += 1
            continue

        for b in _parse_buildings(data):
            mask_name = b["mask_name"]
            if mask_name is None:
                continue
            mask_path = mask_dir / mask_name
            if not mask_path.exists():
                n_skipped_missing_files += 1
                continue

            bbox = _mask_bbox(mask_path)
            if bbox is None:
                n_skipped_empty_mask += 1
                continue
            x0, y0, x1, y1 = bbox

            rows.append({
                "building_id":  b["building_id"],
                "frame_name":   frame_stem,
                "frame_path":   str(frame_path.resolve()),
                "mask_name":    mask_name,
                "mask_path":    str(mask_path.resolve()),
                "damage_level": b["damage_level"],
                "bbox_x0": x0, "bbox_y0": y0,
                "bbox_x1": x1, "bbox_y1": y1,
                "bbox_w": x1 - x0, "bbox_h": y1 - y0,
                "dataset_group": frame_stem.split("_")[0],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No buildings parsed — check the raw_dir structure.")

    # Frame-level split
    splits = _frame_level_split(df["frame_name"].tolist(),
                                train_frac=train_frac,
                                val_frac=val_frac,
                                seed=seed)
    df["split"] = df["frame_name"].map(splits)

    if n_skipped_missing_files:
        print(f"  Skipped {n_skipped_missing_files} entries with missing frame/mask files")
    if n_skipped_empty_mask:
        print(f"  Skipped {n_skipped_empty_mask} entries with empty masks")

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

    return df


# ---------------------------------------------------------------------------
# Quick description helper
# ---------------------------------------------------------------------------

def describe(df: pd.DataFrame) -> None:
    print(f"Total buildings: {len(df):,}")
    print(f"Total frames:    {df['frame_name'].nunique()}")
    print(f"\nBuildings per damage level:")
    print(df["damage_level"].value_counts().sort_index().to_string())
    print(f"\nBuildings per split:")
    print(df["split"].value_counts().to_string())
    print(f"\nBuildings per (split × level):")
    print(pd.crosstab(df["split"], df["damage_level"]).to_string())
    print(f"\nBbox sizes (pixels):")
    print(df[["bbox_w", "bbox_h"]].describe().round(1).to_string())
