# Runoff Forecasting: Deep Learning Post-processing of NWM

Final project scaffold for Task 1 of the AI course final. Post-processes NOAA
National Water Model (NWM) short-range forecasts using sequence models so the
corrected forecast better matches USGS observed runoff.

**Done (infrastructure + all three models):**
- Data ingestion and alignment (NWM monthly CSVs + USGS 15-min → hourly parquet)
- Sliding-window PyTorch dataset with proper fit-on-train-only scaling
- Standard hydrology metrics (NSE, KGE, RMSE, MAE, PBIAS)
- Training loop with early stopping
- Evaluation comparing DL-corrected forecast vs raw NWM on the held-out test set
- Three required deep learning models: **LSTM**, **GRU**, **Transformer**
- Comparison sweep script that trains every combination and builds a report-ready table

**Left to do:**
- Plug in **ERA5** meteorological features (preprocessing has a clean hook)
- Run the full comparison sweep, inspect results, tune hyperparameters

---

## Project layout

```
runoff_forecasting/
├── src/
│   ├── preprocessing.py      NWM+USGS → tidy hourly parquet
│   ├── dataset.py            Sliding-window PyTorch Dataset, residual target
│   ├── models.py             LSTM, GRU, Transformer (all implemented)
│   ├── metrics.py            NSE, KGE, RMSE, MAE, PBIAS
│   └── train.py              Training loop + test evaluation
├── scripts/
│   ├── preprocess.py         CLI: raw dir → parquet files
│   ├── train_baseline.py     CLI: train one model, save artifacts
│   └── run_comparison.py     CLI: sweep across models/gauges/leads, build table
├── notebooks/
│   └── 01_eda.ipynb          Exploratory data analysis (run after preprocessing)
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

## Step 1.5 — Exploratory data analysis

Open `notebooks/01_eda.ipynb` in Jupyter or VSCode and run all cells:

```bash
jupyter notebook notebooks/01_eda.ipynb
# or just open it in VSCode and click "Run All"
```


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

## Step 3 — Train the GRU and Transformer

All three required models are now implemented in `src/models.py` (LSTM, GRU,
TransformerPredictor). You can train them individually via `train_baseline.py`
with `--model gru` or `--model transformer`:

```bash
python scripts/train_baseline.py --parquet data/processed/gauge_21609641.parquet --model gru --target-lead 6 --epochs 30
python scripts/train_baseline.py --parquet data/processed/gauge_21609641.parquet --model transformer --target-lead 6 --epochs 30
```

Expected parameter counts (for the report):
- LSTM ≈ 59,265 parameters
- GRU ≈ 45,505 parameters (22% fewer — 3 gates instead of 4)
- Transformer ≈ 72,449 parameters

## Step 4 — Run the full model comparison

Rather than running each combination by hand, use `run_comparison.py`:

```bash
# Full sweep: both gauges × 4 lead times × 3 models = 24 runs
# Expect 30-60 minutes on a laptop CPU.
python scripts/run_comparison.py --leads 1 6 12 18 --epochs 30
```