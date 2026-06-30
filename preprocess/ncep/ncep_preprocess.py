import xarray as xr
import os
import multiprocessing as mp
from fnmatch import fnmatch
from tqdm import tqdm
import time
import shutil
from config_loader import CONFIG

def RoundBase(x, prec=0, base=1):
    return round(base * round(float(x)/base),prec)

def preprocess_ncep(file, output_path):
    # Load dataset
    paths = [file]
    ds_list = []
    extraction_info = [
        {
            "typeOfLevel": "isobaricInhPa",
            "cfVarName_list": ["u", "v", "w", "absv", "t", "gh", "r"],
            "renameVars": {"u": "U", "v": "V", "w": "OMEGA", "absv": "EPV", "t": "T", "gh": "H", "r": "RH"}
        },
        {
            "typeOfLevel": "surface",
            "cfVarName_list": ["t", "sp", "lsm"],
            "renameVars": {"t": "TSK", "sp": "PS", "lsm": "landmask"}
        },
        {
            "typeOfLevel": "tropopause",
            "cfVarName_list": ["gh", "t"],
            "renameVars": {"gh": "hgttrp", "t": "tmptrp"}
        },
    ]
    
    for ex_i in extraction_info:
        ds_l = []
        for varname in ex_i["cfVarName_list"]:
            ds_raw = [
                xr.open_dataset(f, engine="cfgrib", backend_kwargs={
                    "filter_by_keys": {
                        "cfVarName": varname, "typeOfLevel": ex_i["typeOfLevel"]
                    },
                    "indexpath": ""
                })
                for f in paths
            ]
            ds_raw = [
                d.drop([v for v in (list(d.coords) + list(d.data_vars))
                        if v not in (list(d.indexes) + list(d.keys()) + ["time", "step"])])
                for d in ds_raw
            ]
            ds_l.append(xr.combine_nested(ds_raw, concat_dim="time"))
        ds_tmp = xr.merge(ds_l)
        if ex_i["renameVars"]:
            ds_tmp = ds_tmp.rename(ex_i["renameVars"])
        ds_list.append(ds_tmp)
    ds = xr.merge(ds_list, compat="override")
    
    ds = ds.drop_vars("step")
    # Crop to region of interest
    ds = ds.where(ds.latitude <= CONFIG.PRE_DOMAIN.MAX_LAT, drop=True)
    ds = ds.where(ds.latitude >= CONFIG.PRE_DOMAIN.MIN_LAT, drop=True)
    ds = ds.where(ds.longitude <= CONFIG.PRE_DOMAIN.MAX_LON, drop=True)
    ds = ds.where(ds.longitude >= CONFIG.PRE_DOMAIN.MIN_LON, drop=True)
    # Sort data values
    ds = ds.sortby("isobaricInhPa", ascending=False)

    # Save to disk
    filename = os.path.basename(file) # fnl_20101130_18_00.grib2 or fnl_20101130_18_00.grib1
    filename = filename.replace(".grib2", "").replace(".grib1", "") # fnl_20101130_18_00
    save_path = os.path.join(output_path, f"{filename}.nc")
    ds.to_netcdf(save_path)

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
        preprocess_ncep(file, output_path)
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
        

def PreprocessNcepMain():
    print("Preprocess Ncep")
    input_path = CONFIG.IPATH.NCEP_RAW
    output_path = CONFIG.OPATH.NCEP_PREP

    CleanDir(output_path)
    files = RecurseListDir(input_path, ["*.grib1", "*.grib2"])
    queue = mp.Queue()
    for f in files:
        queue.put(f)  
    mp.Process(target=ProgressBar, args=(queue, len(files), "Preprocess Ncep")).start()
    MultiProcessing(Worker, (queue, output_path), 32)
    print("Preprocess Ncep: Done")

if __name__ == "__main__":
    PreprocessNcepMain()
    print("Preprocess Ncep: Completed.")
