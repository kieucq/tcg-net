import xarray as xr
import os
import multiprocessing as mp
from fnmatch import fnmatch
from tqdm import tqdm
import time
import shutil
import numpy as np
import pandas as pd
from config_loader import CONFIG

MODEL_TO_ERA5_VARIABLE = {
    "PHIS": "BLH",
    "PS": "SP",
    "SLP": "MSL",
    "H": "Z",
    "OMEGA": "W",
    "QL": "QC",
    "QV": "Q",
    "RH": "R",
    "SST": "SSTK"
}


def configured_model_variables():
    model_config = CONFIG.DYNAMIC_MODEL_DATASET
    variables = list(model_config.SINGLE_VAR) + list(model_config.PRESS_VAR)
    if len(variables) != len(set(variables)):
        raise ValueError(
            "DYNAMIC_MODEL_DATASET.SINGLE_VAR and PRESS_VAR contain duplicate variables"
        )
    return variables


def select_configured_variables(dataset):
    """Select and standardize exactly the model variables requested in config.json."""
    configured_variables = configured_model_variables()
    source_variables = []
    rename_variables = {}
    missing_variables = []

    for model_name in configured_variables:
        raw_name = MODEL_TO_ERA5_VARIABLE.get(model_name, model_name)
        if model_name in dataset.data_vars:
            source_name = model_name
        elif raw_name in dataset.data_vars:
            source_name = raw_name
        else:
            missing_variables.append(f"{model_name} (expected ERA5 variable {raw_name})")
            continue

        source_variables.append(source_name)
        if source_name != model_name:
            rename_variables[source_name] = model_name

    if missing_variables:
        raise ValueError(
            "ERA5 input is missing variables required by config.json: "
            + ", ".join(missing_variables)
        )

    selected = dataset[source_variables]
    if rename_variables:
        selected = selected.rename(rename_variables)
    return selected[configured_variables]


def RoundBase(x, prec=0, base=1):
    return round(base * round(float(x)/base),prec)

def preprocess_era5(file, output_path):
    ds = xr.open_dataset(file, engine="netcdf4")

    if "level" not in ds.dims:
        raise ValueError(f"ERA5 pressure dimension 'level' is missing from {file}")
    ds_filtered = ds.isel(level=slice(None, None, -1))
    ds_filtered['level'].attrs.update(
        {
            "stored_direction":"increasing",
            "positive":"up"
        }
    )
    ds_filtered = select_configured_variables(ds_filtered)
    ds_filtered = ds_filtered.rename({"level": "isobaricInhPa"})

    lat_vals = np.arange(CONFIG.PRE_DOMAIN.MIN_LAT, CONFIG.PRE_DOMAIN.MAX_LAT + 0.1, 0.5)  # from -50 to 70, step 0.5
    lon_vals = np.arange(CONFIG.PRE_DOMAIN.MIN_LON, CONFIG.PRE_DOMAIN.MAX_LON + 0.1, 0.5)  # from 60 to 220, step 0.5

    ds_filtered = ds_filtered.sel(
        latitude=lat_vals,
        longitude=lon_vals,
        method="nearest"
    )

    # ds_filtered = ds_filtered.squeeze("time", drop=True)

    # Change axis of longitude
    # lon_original = ds_filtered["longitude"]
    # lon_normalized = [RoundBase(
    #     lon + 360, prec=1, base=0.5) if lon < 0 else lon for lon in lon_original]
    # ds_filtered["longitude"] = lon_normalized
    # # Fix latitude
    # lat_original = ds_filtered["latitude"]
    # lat_normalized = [RoundBase(lat, prec=1, base=0.5)
    #                     for lat in lat_original]
    # ds_filtered["latitude"] = lat_normalized
    
    # Save processed dataset
    # output_name = file.split('/')[-1]
    timestamp = pd.Timestamp(ds['time'].values[0]).to_pydatetime()
    output_name = timestamp.strftime("era5_%Y%m%d_%H_%M.nc")
    output_name = os.path.join(output_path, output_name)
    ds_filtered.to_netcdf(output_name)

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
    failed_workers = [p for p in ps if p.exitcode != 0]
    if failed_workers:
        failures = ", ".join(
            f"pid={p.pid}, exitcode={p.exitcode}" for p in failed_workers
        )
        raise RuntimeError(f"ERA5 preprocessing workers failed: {failures}")

def Worker(queue: mp.Queue, output_path: str):
    while not queue.empty():
        file = queue.get()
        if not file:
            break
        preprocess_era5(file, output_path)
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
        

def PreprocessEra5Main():
    print("Preprocess Era5")
    input_path = CONFIG.IPATH.ERA5_RAW
    output_path = CONFIG.OPATH.ERA5_PREP

    files = RecurseListDir(input_path, ["*.nc"])
    if not files:
        raise FileNotFoundError(f"No ERA5 NetCDF files were found under {input_path}")

    # Fail before deleting prior output or starting workers when the raw-data
    # variables do not agree with config.json.
    with xr.open_dataset(files[0], engine="netcdf4") as sample:
        if "level" not in sample.dims:
            raise ValueError(f"ERA5 pressure dimension 'level' is missing from {files[0]}")
        select_configured_variables(sample)

    CleanDir(output_path)
    queue = mp.Queue()
    for f in files:
        queue.put(f)  
    progress_process = mp.Process(
        target=ProgressBar,
        args=(queue, len(files), "Preprocess Era5"),
    )
    progress_process.start()
    try:
        MultiProcessing(Worker, (queue, output_path), 32)
    finally:
        if progress_process.is_alive():
            progress_process.terminate()
        progress_process.join()
    print("Preprocess Era5: Done")

if __name__ == "__main__":
    PreprocessEra5Main()
    print("Preprocess Era5: Completed.")
