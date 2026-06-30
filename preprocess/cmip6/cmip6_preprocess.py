import xarray as xr
import os
import multiprocessing as mp
from fnmatch import fnmatch
from tqdm import tqdm
import time
import shutil
import pandas as pd
import numpy as np
from config_loader import CONFIG

def RoundBase(x, prec=0, base=1):
    return round(base * round(float(x)/base),prec)

def preprocess_cmip6(file, output_path):
    # Load dataset
    paths = [file]
    ds_list = [xr.load_dataset(
            filename_or_obj=f, engine="netcdf4") for f in paths]
    ds_tmp = xr.merge(ds_list)
    # Rename vars to syncs with other datasets
    ds = ds_tmp.rename({
            "south_north": "latitude",
            "west_east": "longitude",
            "bottom_top": "isobaricInhPa",
            "Time": "time",
            "XLAT": "latitude",
            "XLONG": "longitude",
            "XTIME": "time"
        })
    
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
        lon + 360, prec=1, base=0.1) if lon < 0 else lon for lon in lon_flatten]
    ds = ds.assign(longitude=(('longitude'), lon_normalized))

    # Fix latitude
    lat_original = ds["latitude"].values
    # Decrease dimension of longtitude from 3D to 1D array
    lat_flatten = lat_original.flatten()
    lat_flatten = np.unique(lat_flatten)
    lat_normalized = [RoundBase(lat, prec=1, base=0.1)
                        for lat in lat_flatten]
    ds = ds.assign(latitude=(('latitude'), lat_normalized))

    # Drop the unnecessary variables
    ds = ds.drop_vars(['XLAT_U', 'XLONG_U', 'XLAT_V', 'XLONG_V', 'PH', 'PHB'])
    # Crop to region of interest
    ds = ds.where(ds.latitude <= CONFIG.PRE_DOMAIN.MAX_LAT, drop=True)
    ds = ds.where(ds.latitude >= CONFIG.PRE_DOMAIN.MIN_LAT, drop=True)
    ds = ds.where(ds.longitude <= CONFIG.PRE_DOMAIN.MAX_LON, drop=True)
    ds = ds.where(ds.longitude >= CONFIG.PRE_DOMAIN.MIN_LON, drop=True)
    # Set the coordinates
    ds.set_coords(['latitude', 'longitude', 'time'])
    
    # Sort data values
    ds = ds.sortby("longitude")
    ds = ds.sortby("latitude")
    ds = ds.sortby("time")

    timestamp = pd.Timestamp(ds['time'].values[0]).to_pydatetime()
    filename = timestamp.strftime("%Y%m%d_%H_%M")
    filepath_to_save = os.path.join(output_path, f"cmip6_{filename}.nc")
    ds.to_netcdf(filepath_to_save)

def RecurseListDir(root: str, pattern: list[str]):
    f = []
    for p in pattern:
        for path, subdirs, files in os.walk(root):
            for name in files:
                if fnmatch(name, p):
                    f.append(os.path.join(path, name))
    return f

def CleanDir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        return
    for file_obj in os.listdir(path):
        file_obj_path = os.path.join(path, file_obj)
        if (os.path.isfile(file_obj_path)) or (os.path.islink(file_obj_path)):
            os.unlink(file_obj_path)
        else:
            shutil.rmtree(file_obj_path)

def MultiProcessing(Worker, args:tuple, n_worker:int):
    print(f"MultiProcess: {n_worker}")
    ps = []
    for i in range(n_worker):
        p = mp.Process(
            target=Worker,
            args=args
        )
        p.start()
        ps.append(p)
    for p in ps:
        p.join()

def Worker(queue: mp.Queue, output_path: str):
    while not queue.empty():
        file = queue.get()
        if not file:
            break
        preprocess_cmip6(file, output_path)
    print("Done.")
    exit()
    
def ProgressBar(queue: mp.Queue, total:int, name:str="Progress"):
    with tqdm(total = total) as pbar:
        pbar.set_description(name)
        while True:
            current = queue.qsize()
            pbar.n = total - current
            pbar.refresh()
            time.sleep(1)
            if queue.empty():
                pbar.close()
                exit()
        

def PreprocessCmip6Main():
    print("Preprocess Cmip6")
    input_path = CONFIG.IPATH.CMIP6_RAW
    output_path = CONFIG.OPATH.CMIP6_PREP

    CleanDir(output_path)
    files = RecurseListDir(input_path, ["*"])
    queue = mp.Queue()
    for f in files:
        queue.put(f)  
    mp.Process(target=ProgressBar, args=(queue, len(files), "Preprocess Cmip6")).start()
    MultiProcessing(Worker, (queue, output_path), 32)
    print("Preprocess Cmip6: Done")

if __name__ == "__main__":
    PreprocessCmip6Main()
    print("Preprocess Cmip6: Completed.")
