# Runoff Forecasting: Deep Learning Post-processing of NWM

Final project scaffold for Task 1 of the AI course final. Post-processes NOAA
National Water Model (NWM) short-range forecasts using sequence models so the
corrected forecast better matches USGS observed runoff.

## What's here vs. what you need to do

**Done for you (infrastructure):**
- Data ingestion and alignment (NWM monthly CSVs + USGS 15-min → hourly parquet)
- Sliding-window PyTorch dataset with proper scaler fit-on-train-only
- Standard hydrology metrics (NSE, KGE, RMSE, MAE, PBIAS)
- Training loop with early stopping
- Evaluation comparing DL-corrected forecast vs raw NWM on the held-out test set
- A working **LSTM** baseline

**Left for you (the actual assignment):**
- Implement `GRUPredictor` in `src/models.py` (stubbed with hints)
- Implement `TransformerPredictor` in `src/models.py` (stubbed with hints)
- Plug in **ERA5** meteorological features (preprocessing has a clean hook)
- Run experiments across lead times and gauges, tune hyperparameters
- Write the 5-page report and 15-minute presentation

---

## Project layout

```
runoff_forecasting/
├── src/
│   ├── preprocessing.py      NWM+USGS → tidy hourly parquet
│   ├── dataset.py            Sliding-window PyTorch Dataset, residual target
│   ├── models.py             LSTM (done), GRU + Transformer (your job)
│   ├── metrics.py            NSE, KGE, RMSE, MAE, PBIAS
│   └── train.py              Training loop + test evaluation
├── scripts/
│   ├── preprocess.py         CLI: raw dir → parquet files
│   └── train_baseline.py     CLI: train one model, save artifacts
├── data/
│   └── processed/            (parquets land here)
├── runs/                     (created per training run)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Step 1 — Preprocess the raw data

Point the script at wherever the raw midterm folder lives:

```bash
python scripts/preprocess.py \
    --raw-dir path/to/RunoffForcastingProject
```

This expects the familiar layout with one subdirectory per gauge reach
(`20380357/`, `21609641/`), each containing the monthly
`streamflow_<reach>_YYYYMM.csv` files and the USGS observation CSV.

Output: `data/processed/gauge_<reach>.parquet` per gauge. Each parquet is
~17k rows of aligned hourly data with these columns:

| column | meaning |
|---|---|
| `usgs_flow` | USGS observation, hourly mean of the 15-min series |
| `usgs_estimated` | True if any 15-min value in that hour was flagged `e` |
| `nwm_lead_1..nwm_lead_18` | NWM forecast valid at this hour, made at `t - lead` hours |
| `resid_lead_1..resid_lead_18` | `usgs_flow - nwm_lead_h` — the residual the model is trained to predict |

## Step 2 — Train the LSTM baseline

```bash
python scripts/train_baseline.py \
    --parquet data/processed/gauge_21609641.parquet \
    --model lstm \
    --target-lead 6 \
    --epochs 30
```

Key flags:
- `--target-lead 1..18` — which forecast lead time to correct
- `--lookback 48` — hours of history fed to the model
- `--target-kind residual|flow` — predict the residual (default, Han & Morrison
  framing) or the flow directly
- `--hidden-size`, `--num-layers`, `--dropout`, `--lr`, `--batch-size`, `--epochs`

Artifacts land in `runs/<model>_lead<h>_<timestamp>/`:
- `model.pt` — best-val-loss weights
- `history.csv` — per-epoch train/val loss
- `test_predictions.csv` — timestamp, obs, raw NWM, corrected NWM, predicted residual
- `summary.json` — config, param count, baseline vs DL metrics, improvements

## Step 3 — Implement GRU and Transformer

Open `src/models.py`. The `GRUPredictor` and `TransformerPredictor` classes
currently raise `NotImplementedError` and have docstrings with implementation
hints. **Keep the hidden size / number of layers matched to the LSTM** so the
comparison is about architecture, not capacity.

Once implemented, run:

```bash
python scripts/train_baseline.py --parquet ... --model gru --target-lead 6 --epochs 30
python scripts/train_baseline.py --parquet ... --model transformer --target-lead 6 --epochs 30
```

and compare the three `summary.json` files.

---

## What a "fair" comparison looks like (for the report)

For each (model, gauge, lead time) combination:

1. Same lookback window (48h recommended).
2. Same train/val/test split — the test window (Oct 2022 – Apr 2023) is
   hard-coded in `split_train_val_test` per the assignment rules; never
   touch it during training or tuning.
3. Same random seed. Even better, run each model 3–5 times with different
   seeds and report mean ± std.
4. Match parameter count approximately (hidden size × num layers).
5. Compare on the same metrics, against the same raw-NWM baseline.

Report table you'll want to produce:

| Gauge | Lead (h) | Model | NSE ↑ | KGE ↑ | RMSE ↓ | MAE ↓ | PBIAS → 0 | Params |
|---|---|---|---|---|---|---|---|---|
| 21609641 | 6 | NWM raw | 0.928 | 0.791 | 4.55 | 2.49 | −16.4 | — |
| 21609641 | 6 | LSTM | 0.948 | 0.843 | 3.86 | 1.79 | −10.7 | 59,265 |
| 21609641 | 6 | GRU | ... | ... | ... | ... | ... | ... |
| 21609641 | 6 | Transformer | ... | ... | ... | ... | ... | ... |
| ...and so on for multiple lead times and both gauges |

Good plots for the report:
- Training curves (train vs val loss per epoch, one panel per model).
- Hydrograph over a representative test-period week: obs, raw NWM, corrected.
- Scatter: predicted vs observed flow for raw NWM and for each model.
- NSE (or KGE) vs lead hour for all three models — shows where each does best.

---

## Adding ERA5 (for the final-exam requirement)

ERA5 reanalysis gives you hourly meteorological forcings (precipitation,
2-metre temperature, radiation, wind, etc.) that NWM itself consumes. Adding
them as model inputs is often what pushes DL post-processing from "slightly
better than NWM" to "meaningfully better".

Suggested workflow:

1. Look up each gauge's lat/lon (via the USGS Site Info web service) and
   pick a small spatial window around each basin.
2. Register for a free Copernicus CDS account (`cds.climate.copernicus.eu`).
   Approval can take 1–2 days, so **start this early**.
3. Use the `cdsapi` Python package to pull ERA5-Land hourly variables:
   `total_precipitation`, `2m_temperature`, `surface_solar_radiation_downwards`,
   `10m_u_component_of_wind`, `10m_v_component_of_wind`. Spatially-average
   over the basin to a single time series per variable.
4. Extend `build_gauge_dataset` to left-join those columns onto the hourly
   parquet. Add them to `default_feature_cols()` in `src/dataset.py`.
5. Re-run all three models and compare with/without ERA5 — that
   ablation is great material for the report.

---

## Troubleshooting

- **`ModuleNotFoundError: src`** — run the scripts from the project root
  (`runoff_forecasting/`), not from inside `scripts/` or `src/`.
- **Slow on CPU** — the default (2-layer, 64-hidden LSTM, 48h window, 128
  batch) is ~10s/epoch on CPU, so 30 epochs is ~5 minutes. Fine for laptops.
  On GPU it'll be seconds.
- **Val loss higher than train loss** — expected; we're regularizing with
  dropout. What you want is for val loss to *decrease over epochs*.
- **Negative NSE on test** — raw NWM on gauge 20380357 has an extreme bias
  (NWM mean ~22× USGS mean). The DL correction still dramatically improves
  it, but not enough to reach positive NSE. Worth flagging in the report.

---

## References

Han, H. & Morrison, R. R. (2022). *Improved runoff forecasting performance
through error predictions using a deep-learning approach.* Journal of
Hydrology, 608, 127653. → `Improved runoff forecasting performance
through error predictions using a deep-learning approach.pdf` in the
midterm folder.

NOAA National Water Model: <https://water.noaa.gov/about/nwm>

ERA5-Land docs: <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land>

USGS Water Services: <https://waterservices.usgs.gov/>
