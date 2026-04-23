import xarray as xr

ds1 = xr.open_dataset("data_stream-oper_stepType-instant.nc")
ds2 = xr.open_dataset("data_stream-oper_stepType-accum.nc")

df1 = ds1.to_dataframe().reset_index()
df2 = ds2.to_dataframe().reset_index()

print(df1.head())
print(df2.head())
