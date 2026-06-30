import xarray as xr
import os
import multiprocessing as mp
from fnmatch import fnmatch
from tqdm import tqdm
import time
import shutil
import pandas as pd
import copy
import datetime
from config_loader import CONFIG

def RoundBase(x, prec=0, base=1):
    return round(base * round(float(x)/base),prec)

def preprocess_merra2(file, output_path):
    # Load dataset
    paths = [file]
    ds_list = [xr.load_dataset(
            filename_or_obj=f, engine="netcdf4") for f in paths]
    ds_tmp = xr.merge(ds_list)
    # Rename vars to syncs with other datasets
    ds = ds_tmp.rename({
            "lat": "latitude",
            "lon": "longitude",
            "lev": "isobaricInhPa"
        })
    
    # Change axis of longitude
    lon_original = ds["longitude"]
    lon_normalized = [RoundBase(
        lon + 360, prec=3, base=0.625) if lon < 0 else lon for lon in lon_original]
    ds["longitude"] = lon_normalized
    # Fix latitude
    lat_original = ds["latitude"]
    lat_normalized = [RoundBase(lat, prec=1, base=0.5)
                        for lat in lat_original]
    ds["latitude"] = lat_normalized
    # Crop to region of interest
    ds = ds.where(ds.latitude <= CONFIG.PRE_DOMAIN.MAX_LAT, drop=True)
    ds = ds.where(ds.latitude >= CONFIG.PRE_DOMAIN.MIN_LAT, drop=True)
    ds = ds.where(ds.longitude <= CONFIG.PRE_DOMAIN.MAX_LON, drop=True)
    ds = ds.where(ds.longitude >= CONFIG.PRE_DOMAIN.MIN_LON, drop=True)
    # Sort data values
    ds = ds.sortby("longitude")
    ds = ds.sortby("latitude")
    ds = ds.sortby("time")

    # Split one large dataset to smaller dataset. (one timestamp each dataset).
    ds_list = []
    for t in list(ds.indexes["time"]):
        sub_ds = ds.sel(time=t)
        sub_ds.attrs["ISO_TIME"] = t.strftime("%Y-%m-%d %H:%M:%S")
        ds_list.append(copy.deepcopy(sub_ds))

    for sub_ds in ds_list:
        timestamp = datetime.datetime.fromisoformat(str(sub_ds.attrs["ISO_TIME"]))
        filename = timestamp.strftime("%Y%m%d_%H_%M")
        filepath_to_save = os.path.join(output_path, f"merra2_{filename}.nc")
        sub_ds.to_netcdf(filepath_to_save)

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
        preprocess_merra2(file, output_path)
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
        

def PreprocessMerra2Main():
    print("Preprocess Merra2")
    input_path = CONFIG.IPATH.MERRA2_RAW
    output_path = CONFIG.OPATH.MERRA2_PREP

    CleanDir(output_path)
    files = RecurseListDir(input_path, ["*.nc4"])
    queue = mp.Queue()
    for f in files:
        queue.put(f)  
    mp.Process(target=ProgressBar, args=(queue, len(files), "Preprocess Merra2")).start()
    MultiProcessing(Worker, (queue, output_path), 32)
    print("Preprocess Merra2: Done")

if __name__ == "__main__":
    PreprocessMerra2Main()
    print("Preprocess Merra2: Completed.")
