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

def configured_model_variables():
    model_config = CONFIG.DYNAMIC_MODEL_DATASET
    variables = list(model_config.SINGLE_VAR) + list(model_config.PRESS_VAR)
    if len(variables) != len(set(variables)):
        raise ValueError("DYNAMIC_MODEL_DATASET.SINGLE_VAR and PRESS_VAR contain duplicate variables")
    return variables

def RoundBase(x, prec=0, base=1):
    return round(base * round(float(x)/base),prec)

def preprocess_gfs(file, output_path):
    paths = [file]
    ds_list = []
    TARGET_LEVELS = CONFIG.DYNAMIC_MODEL_DATASET.PRESS_LEVEL

    extraction_info = [
        {
            "typeOfLevel": "isobaricInhPa",
            "cfVarName_list": ["u", "v", "w", "absv", "t", "gh", "r", "q", "o3mr", "icmr", "clwmr"],
            "renameVars": {"u": "U", "v": "V", "w": "OMEGA", "absv": "EPV", "t": "T", "gh": "H", "r": "RH", "q": "QV", "o3mr": "O3", "icmr": "QI", "clwmr": "QL"}
        },
        {
            "typeOfLevel": "surface",
            "cfVarName_list": ["sp", "orog", "t"],
            "renameVars": {"sp": "PS", "orog": "PHIS", "t": "SST"}
        },
        {
            "typeOfLevel": "meanSea",
            "cfVarName_list": ["prmsl"],
            "renameVars": {"prmsl": "SLP"}
        }
    ]

    for ex_i in extraction_info:
        ds_l = []
        for varname in ex_i["cfVarName_list"]:
            ds_raw = []
            for f in paths:
                try:
                    d = xr.open_dataset(f, engine="cfgrib", backend_kwargs={
                        "filter_by_keys": {"shortName": varname, "typeOfLevel": ex_i["typeOfLevel"]},
                        "indexpath": ""
                    })
                    # BẢO VỆ 1: Chỉ lấy nếu dataset thực sự có dữ liệu
                    if len(d.variables) > 0:
                        ds_raw.append(d)
                except Exception:
                    continue 

            if not ds_raw:
                continue # Nếu rỗng thì bỏ qua biến này

            # BẢO VỆ 2: Bỏ biến thừa an toàn, không làm mất tọa độ
            ds_raw_processed = []
            for d in ds_raw:
                safe_vars = list(d.indexes) + list(d.keys()) + ["time", "step", "valid_time", "latitude", "longitude", "isobaricInhPa"]
                drop_list = [v for v in (list(d.coords) + list(d.data_vars)) if v not in safe_vars]
                d = d.drop_vars(drop_list, errors="ignore")
                ds_raw_processed.append(d)
            ds_raw = ds_raw_processed

            if varname == "r":
                ds_raw = [d / 100.0 for d in ds_raw]

            # BẢO VỆ 3: Chỉ nội suy nếu trục isobaricInhPa thực sự tồn tại
            if ex_i["typeOfLevel"] == "isobaricInhPa":
                ds_raw_processed = []
                for d in ds_raw:
                    if "isobaricInhPa" in d.dims or "isobaricInhPa" in d.coords:
                        d = d.sortby("isobaricInhPa", ascending=False)
                        d = d.reindex(isobaricInhPa=TARGET_LEVELS, method="ffill")
                    ds_raw_processed.append(d)
                ds_raw = ds_raw_processed

            if ds_raw:
                ds_l.append(xr.combine_nested(ds_raw, concat_dim="time"))

        if ds_l:
            ds_tmp = xr.merge(ds_l, compat="override")
            if ex_i["typeOfLevel"] == "isobaricInhPa" and "isobaricInhPa" in ds_tmp.coords:
                ds_tmp['isobaricInhPa'].attrs.update({"stored_direction":"increasing", "positive":"up"})

            if ex_i["renameVars"]:
                ds_tmp = ds_tmp.rename({k: v for k, v in ex_i["renameVars"].items() if k in ds_tmp.data_vars})

            ds_list.append(ds_tmp)

    if not ds_list:
        return # Thoát luôn nếu file hoàn toàn rỗng

    ds = xr.merge(ds_list, compat="override")

    if "step" in ds.coords or "step" in ds.data_vars:
        ds = ds.drop_vars("step", errors="ignore")

    configured_vars = configured_model_variables()
    vars_to_keep = [v for v in configured_vars if v in ds.data_vars]
    
    if not vars_to_keep:
        return 
        
    ds = ds[vars_to_keep]

    lat_vals = np.arange(CONFIG.PRE_DOMAIN.MIN_LAT, CONFIG.PRE_DOMAIN.MAX_LAT + 0.1, 0.5)
    lon_vals = np.arange(CONFIG.PRE_DOMAIN.MIN_LON, CONFIG.PRE_DOMAIN.MAX_LON + 0.1, 0.5)

    if "latitude" in ds.coords and "longitude" in ds.coords:
        ds = ds.sel(latitude=lat_vals, longitude=lon_vals, method="nearest")

    if "isobaricInhPa" in ds.coords:
        ds = ds.sortby("isobaricInhPa", ascending=False)

    if 'valid_time' in ds.coords or 'valid_time' in ds.data_vars:
        time_val = ds['valid_time'].values
    else:
        time_val = ds['time'].values[0] if ds['time'].ndim > 0 else ds['time'].values

    timestamp = pd.Timestamp(time_val).to_pydatetime()
    output_name = timestamp.strftime("gfs_%Y%m%d_%H_%M.nc")
    save_path = os.path.join(output_path, output_name)

    ds_trimmed = ds.isel(time=0) if "time" in ds.dims else ds
    ds_trimmed.to_netcdf(save_path)


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
        p = mp.Process(target=Worker, args=args)
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
    failed_workers = [p for p in ps if p.exitcode != 0]
    if failed_workers:
        failures = ", ".join(f"pid={p.pid}, exitcode={p.exitcode}" for p in failed_workers)
        raise RuntimeError(f"GFS preprocessing workers failed: {failures}")


def Worker(queue: mp.Queue, output_path: str):
    while not queue.empty():
        file = queue.get()
        if not file:
            break
        preprocess_gfs(file, output_path)
    exit()


def ProgressBar(queue: mp.Queue, total:int, name:str="Progress"):
    with tqdm(total=total) as pbar:
        pbar.set_description(name)
        while True:
            current = queue.qsize()
            pbar.n = total - current
            pbar.refresh()
            time.sleep(1)
            if queue.empty():
                pbar.close()
                break


def PreprocessGfsMain():
    print("Preprocess GFS: Start")
    input_path = CONFIG.IPATH.GFS_RAW
    output_path = CONFIG.OPATH.GFS_PREP

    files = RecurseListDir(input_path, ["*.grib1", "*.grib2", "*.f000", "*.grb2", "*.grb"])
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file GRIB nào trong {input_path}")

    CleanDir(output_path)
    queue = mp.Queue()
    for f in files:
        queue.put(f)

    # Dùng daemon=True để tránh Progress bar bị treo nếu Worker chết
    p_bar = mp.Process(target=ProgressBar, args=(queue, len(files), "Preprocess GFS"))
    p_bar.daemon = True 
    p_bar.start()

    MultiProcessing(Worker, (queue, output_path), 16)
    print("Preprocess GFS: Done")

if __name__ == "__main__":
    PreprocessGfsMain()
    print("Preprocess GFS: Completed.")
