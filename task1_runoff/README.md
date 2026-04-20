# Runoff Forecasting: Deep Learning Post-processing of NWM

Final project scaffold for Task 1 of the AI course final. Post-processes NOAA
National Water Model (NWM) short-range forecasts using sequence models so the
corrected forecast better matches USGS observed runoff.

## What's here vs. what you need to do

**Done (infrastructure):**
- Data ingestion and alignment (NWM monthly CSVs + USGS 15-min → hourly parquet)
- Sliding-window PyTorch dataset with proper scaler fit-on-train-only
- Standard hydrology metrics (NSE, KGE, RMSE, MAE, PBIAS)
- Training loop with early stopping
- Evaluation comparing DL-corrected forecast vs raw NWM on the held-out test set
- A working **LSTM** baseline

**Left (the actual assignment):**
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

## Step 1.5 — Exploratory data analysis

Open `notebooks/01_eda.ipynb` in Jupyter or VSCode and run all cells:

```bash
jupyter notebook notebooks/01_eda.ipynb
# or just open it in VSCode and click "Run All"
```

The notebook has ten sections covering streamflow characterization, NWM
error analysis, and feature correlation. Each section has markdown prompts
("Your notes here...") where you should record your observations as you go
— these directly feed the Data section of the report.

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

