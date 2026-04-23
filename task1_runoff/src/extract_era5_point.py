import xarray as xr
import pandas as pd

# YOUR GAUGE (11266500)
GAUGE_LAT = 37.71627777777778
GAUGE_LON = -119.66566666666667

# ERA5 uses 0–360 longitude
if GAUGE_LON < 0:
    GAUGE_LON = 360 + GAUGE_LON

ds_instant = xr.open_dataset("data_stream-oper_stepType-instant.nc")
ds_accum = xr.open_dataset("data_stream-oper_stepType-accum.nc")

point_instant = ds_instant.sel(latitude=GAUGE_LAT, longitude=GAUGE_LON, method="nearest")
point_accum = ds_accum.sel(latitude=GAUGE_LAT, longitude=GAUGE_LON, method="nearest")

df_instant = point_instant.to_dataframe().reset_index()
df_accum = point_accum.to_dataframe().reset_index()

df = pd.merge(
    df_instant,
    df_accum[["valid_time", "tp"]],
    on="valid_time",
    how="inner"
)

# keep only needed columns
df = df[["valid_time", "t2m", "sp", "u10", "v10", "tp"]].copy()
df = df.rename(columns={"valid_time": "time"})

# feature engineering
df["t2m_c"] = df["t2m"] - 273.15
df["tp_mm"] = df["tp"] * 1000
df["wind_speed"] = (df["u10"]**2 + df["v10"]**2) ** 0.5

df = df[["time", "t2m_c", "tp_mm", "sp", "wind_speed"]]
df = df.sort_values("time")

df.to_csv("era5_point_timeseries.csv", index=False)

print("saved era5_point_timeseries.csv")
print(df.head())
