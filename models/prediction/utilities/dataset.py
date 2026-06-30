import xarray as xr
import numpy as np
import datetime

def LoadFromDisk(path: str) -> xr.Dataset:
    dataset = xr.load_dataset(filename_or_obj=path, engine="netcdf4")
    return dataset

def SaveToDisk(path: str, dataset: xr.Dataset) -> int:
    try:
        dataset.to_netcdf(path=path)
        return 0
    except Exception as e:
        print(f"Error saving dataset to {path}: {e}")
        return e
    
def FindNearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def SelectData(
        dataset: xr.Dataset,
        lat_range: tuple[float, float], lon_range: tuple[float, float],
        datetime_range: list[tuple[datetime.date, datetime.time]]
    ) -> xr.Dataset:
        ds = dataset
        # Crop data by latitude and longitude
        ds = ds.where(ds.latitude <= lat_range[1], drop=True)
        ds = ds.where(ds.latitude >= lat_range[0], drop=True)
        ds = ds.where(ds.longitude <= lon_range[1], drop=True)
        ds = ds.where(ds.longitude >= lon_range[0], drop=True)
        # Crop data by time
        ds_sel = ds.where(
            ds.time.dt.date.isin([dt[0] for dt in datetime_range]) &
            ds.time.dt.time.isin([dt[1] for dt in datetime_range]),
            drop=True
        )
        # Check if selected data is empty or not
        if (all(not len(ds_sel[data_var].values) for data_var in list(ds_sel.keys()))):
            return None
        return ds_sel
    
def GetSample(
        wds: xr.Dataset, 
        id: str, 
        lat_c: float, 
        lon_c: float, 
        date_c: datetime.date, 
        time_c: datetime.time, 
        lat_dim: int = 0, 
        lon_dim: int = 0, 
        lat_step: float = 0, 
        lon_step: float = 0
    ):
    # LAT CALC
    lat_c_o = lat_c
    lat_grid = np.asarray(wds["latitude"].values)
    lat_c = FindNearest(array=lat_grid, value=lat_c)
    # LON CALC
    lon_c_o = lon_c
    lon_grid = np.asarray(wds["longitude"].values)
    lon_c = FindNearest(array=lon_grid, value=lon_c)
    dt_r = [(date_c, time_c)]
    lat_dim = int(lat_dim/2)
    lon_dim = int(lon_dim/2)
    lat_r = (lat_c-lat_dim*lat_step, lat_c+lat_dim*lat_step)
    lon_r = (lon_c-lon_dim*lon_step, lon_c+lon_dim*lon_step)
    # print(f"GetSample: lat_c={lat_c}, lon_c={lon_c}, lat_r={lat_r}, lon_r={lon_r}, date_c={date_c}, time_c={time_c}, lat_dim={lat_dim}, lon_dim={lon_dim}")
    # Extract dataset
    s_wds = SelectData(wds, lat_r, lon_r, dt_r)
    if not s_wds:
        return None
    # Add metadata to dataset
    s_wds.attrs["SID"] = id
    s_wds.attrs["TYPE"] = "SEQUENCE_AREA"
    s_wds.attrs["ISO_TIME"] = f"{date_c.strftime('%Y-%m-%d')} {time_c.strftime('%H:%M:%S')}"
    s_wds.attrs["LAT"] = f"original={lat_c_o} center={lat_c} min={min(s_wds['latitude'].values)} max={max(s_wds['latitude'].values)}"
    s_wds.attrs["LON"] = f"original={lon_c_o} center={lon_c} min={min(s_wds['longitude'].values)} max={max(s_wds['longitude'].values)}"
    return s_wds