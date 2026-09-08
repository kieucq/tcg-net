# This file extends nasa_merra2.py and written to process wrf_cmip6 raw data
import xarray as xr

from .nasa_merra2 import NasaMerra2
from .__prototype__ import *

class WrfCmip6(NasaMerra2):
    def __init__(self, min_lat: float = -50, max_lat: float = 70, step_lat: float = 0.1, dim_lat: int = 160, min_lon: float = 60, max_lon: float = 220, step_lon: float = 0.1, dim_lon: int = 160, step_time_hours:float = 3.0):
        super().__init__(min_lat, max_lat, step_lat, dim_lat, min_lon, max_lon, step_lon, dim_lon, step_time_hours)
        #### CUSTOM CONSTANTS ####
        self.RENAME_VARS = {
            "south_north": "latitude",
            "west_east": "longitude",
            "bottom_top": "isobaricInhPa",
            "Time": "time",
            "XLAT": "latitude",
            "XLONG": "longitude",
            "XTIME": "time"
        }

    def ProcessRaw(self, dataset: xr.Dataset) -> xr.Dataset:
        ds = dataset
        # Normilize PH, PHB, U and V values
        ph_original = ds['PH']
        ph = (ph_original[:, :-1, :, :] + ph_original[:, 1:, :, :]) / 2
        ds = ds.assign(PH=(('time', 'isobaricInhPa', 'latitude', 'longitude'), ph.data))
        phb_original = ds['PHB']
        phb = (phb_original[:, :-1, :, :] + phb_original[:, 1:, :, :]) / 2
        ds = ds.assign(PHB=(('time', 'isobaricInhPa', 'latitude', 'longitude'), phb.data))
        v = ds['V']
        v = (v[:, :, :-1, :] + v[:, :, 1:, :]) / 2
        ds = ds.assign(V=(('time', 'isobaricInhPa', 'latitude', 'longitude'), v.data))
        u = ds['U']
        u = (u[:, :, :, :-1] + u[:, :, :, 1:]) / 2
        ds = ds.assign(U=(('time', 'isobaricInhPa', 'latitude', 'longitude'), u.data))

        # Calaculate PHIS from PH and PHB
        phis = ds['PH'] + ds['PHB']
        ds = ds.assign(PHIS=(('time', 'isobaricInhPa', 'latitude', 'longitude'), phis.data))

        # Change axis of longitude
        lon_original = ds["longitude"].values
        # Decrease dimension of longtitude from 3D to 1D array
        lon_flatten = lon_original.flatten()
        lon_flatten = np.unique(lon_flatten)
        lon_normalized = [RoundBase(
            lon + 360, prec=1, base=self.STEP_LON) if lon < 0 else lon for lon in lon_flatten]
        ds = ds.assign(longitude=(('longitude'), lon_normalized))

        # Fix latitude
        lat_original = ds["latitude"].values
        # Decrease dimension of longtitude from 3D to 1D array
        lat_flatten = lat_original.flatten()
        lat_flatten = np.unique(lat_flatten)
        lat_normalized = [RoundBase(lat, prec=1, base=self.STEP_LAT)
                          for lat in lat_flatten]
        ds = ds.assign(latitude=(('latitude'), lat_normalized))

        # Drop the unnecessary variables
        ds = ds.drop_vars(['XLAT_U', 'XLONG_U', 'XLAT_V', 'XLONG_V', 'PH', 'PHB'])

        # Crop to region of interest
        ds = ds.where(ds.latitude <= self.MAX_LAT, drop=True)
        ds = ds.where(ds.latitude >= self.MIN_LAT, drop=True)
        ds = ds.where(ds.longitude <= self.MAX_LON, drop=True)
        ds = ds.where(ds.longitude >= self.MIN_LON, drop=True)

        # Set the coordinates
        ds.set_coords(['latitude', 'longitude', 'time'])
        
        # Sort data values
        ds = ds.sortby("longitude")
        ds = ds.sortby("latitude")
        ds = ds.sortby("time")
        
        return ds


    
