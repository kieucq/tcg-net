# This file extends nasa_merra2.py and written to process ERA5 raw data
import xarray as xr

from .nasa_merra2 import NasaMerra2
from .__prototype__ import *

class Era5(NasaMerra2):
    def __init__(self, min_lat: float = -50, max_lat: float = 70, step_lat: float = 0.5, dim_lat: int = 33, min_lon: float = 60, max_lon: float = 220, step_lon: float = 0.5, dim_lon: int = 33, step_time_hours:float = 6.0):
        super().__init__(min_lat, max_lat, step_lat, dim_lat, min_lon, max_lon, step_lon, dim_lon, step_time_hours)
        #### CUSTOM CONSTANTS ####
        self.RENAME_VARS = {
            "level": "isobaricInhPa",
            "BLH": "PHIS",
            "SP": "PS",
            "MSL": "SLP",
            "Z": "H",
            "W": "OMEGA",
            "QC": "QL",
            "Q": "QV",
            "R": "RH",
        }

    def ProcessRaw(self, dataset: xr.Dataset) -> xr.Dataset:
        ds = dataset
        
        # Reverse the level axis to make it in increasing order
        ds_filtered = ds.isel(isobaricInhPa=slice(None, None, -1))
        ds_filtered['isobaricInhPa'].attrs.update(
            {
                "stored_direction":"increasing",
                "positive":"up"
            }
        )

        # Crop to region of interest
        lat_vals = np.arange(-50, 70.1, 0.5)  # from -50 to 70, step 0.5
        lon_vals = np.arange(60, 220.1, 0.5)  # from 60 to 220, step 0.5

        ds_filtered = ds_filtered.sel(
            latitude=lat_vals,
            longitude=lon_vals,
            method="nearest"
        )
        
        return ds_filtered
    
