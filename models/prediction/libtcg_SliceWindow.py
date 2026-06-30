import os
import logging
# logging.basicConfig(format='%(asctime)s %(name)s pid=%(process)-7d | %(levelname)-8s | %(message)s', level=int(os.environ["PREDICT_LOG_LEVEL"]))
# logger = logging.getLogger(
#     name=__name__
# )

import datetime
import numpy as np
import multiprocessing as mp
from utilities.dir import getListDir
from utilities.datetime import convert_datetime64_to_datetime
from utilities.dataset import FindNearest, LoadFromDisk, SaveToDisk, GetSample

def SliceWindow(
    data_type: str, input_path:str, output_path:str,
    lat_min:float, lat_max:float, lon_min:float, lon_max:float,
    lat_dim:float, lon_dim:float, lat_step:float, lon_step:float,
    lat_nstep:int,
    lon_nstep:int,
    proc_count:int,
    subproc_count:int,
    time_to_run: list,
):
    """
    SliceWindow function slices the input weather dataset into smaller windows based on the specified latitude and longitude dimensions.
    """

    queue = mp.Queue()
    # input_file_paths = RecurseListDir(input_path, ["*.nc"])
    input_file_paths = getListDir(input_path, time_to_run, data_type)
    print('len of input file paths', len(input_file_paths))
    # logger.info(f"Total input files: {len(input_file_paths)}")
    for input_file_path in input_file_paths:
        queue.put((
            input_file_path, output_path,
            lat_min, lat_max, lon_min, lon_max,
            lat_dim, lon_dim,
            lat_step, lon_step, lat_nstep, lon_nstep
        ))

    for i in range(proc_count):
        queue.put(None)

    print(f"Total jobs: {queue.qsize()}")
    proc = []
    for i in range(proc_count):
        p = mp.Process(
            target=GetSliceWindow_Worker,
            args=(queue, subproc_count)
        )
        p.start()
        proc.append(p)
    for p in proc:
        p.join()


def GetSliceWindow_Worker(queue:mp.Queue, subproc_count:int):
    """
    Worker function that processes jobs from a queue and calls the GetSliceWindow function.

    Args:
        queue (mp.Queue): The queue containing the jobs to be processed.
        subproc_count (int): The number of sub-processes to be used for saving to NetCDF file.

    Returns:
        None
    """
    # child_logger = logger.getChild("GetSliceWindows_Worker")
    # child_logger.info(f"Start")
    while (queue.qsize()):
        try:
            job = queue.get(timeout=5)
        except:
            continue
        if not (type(job) is tuple):
            break
        # child_logger.debug(f"Remaining queue size: {queue.qsize()}")
        input_file_path, output_path, lat_min, lat_max, lon_min, lon_max, lat_dim, lon_dim, lat_step, lon_step, lat_nstep, lon_nstep = job
        GetSliceWindow(
            input_file_path, output_path,
            lat_min, lat_max, lon_min, lon_max,
            lat_dim, lon_dim,
            lat_step, lon_step,
            lat_nstep, lon_nstep,
            proc_count=subproc_count
        )
    # child_logger.info(f"Exit!")
    # exit()

def GetSliceWindow(
    wd_path:str, output_path:str,
    lat_min:float=None, lat_max:float=None, lon_min:float=None, lon_max:float=None,
    lat_dim:float=16, lon_dim:float=16,
    lat_step:float=0.5,
    lon_step:float=0.5,
    lat_nstep:int=1,
    lon_nstep:int=1,
    proc_count:int=1
):
    """
    Extracts a slice window from a weather dataset and saves the slices as NetCDF files.
    """
    
    print(wd_path)
    w_ds = LoadFromDisk(wd_path)
    LAT_ARR = np.asarray(w_ds["latitude"].values)
    LON_ARR = np.asarray(w_ds["longitude"].values)

    lat_min = np.min(LAT_ARR) if lat_min is None else FindNearest(LAT_ARR, lat_min)
    lat_max = np.max(LAT_ARR) if lat_max is None else FindNearest(LAT_ARR, lat_max)
    lon_min = np.min(LON_ARR) if lon_min is None else FindNearest(LON_ARR, lon_min)
    lon_max = np.max(LON_ARR) if lon_max is None else FindNearest(LON_ARR, lon_max)
    
    try:
        date_time = convert_datetime64_to_datetime(w_ds["valid_time"].values)
    except:
        try:
            date_time = convert_datetime64_to_datetime(w_ds["time"][0].values)
        except:
            date_time = convert_datetime64_to_datetime(w_ds["time"].values)
    # print(date_time)
    date_c = date_time.date()
    time_c = date_time.time()

    # Resolve the symlink to get the actual target path
    real_path = os.path.realpath(output_path)

    """
    SAVE PATH: {temp}/{lat}_{lon}/{time}.nc
    """

    queue = mp.Queue()
    for i in range(len(LAT_ARR)):
        for j in range(len(LON_ARR)):
            if not (lat_nstep == 1 and lon_nstep == 1):
                if i%lat_nstep or j%lon_nstep:
                    continue
            lat_c = LAT_ARR[i]
            lon_c = LON_ARR[j]
            if lat_c > lat_max or lon_c > lon_max or lat_c < lat_min or lon_c < lon_min:
                continue
            sample_path = os.path.join(real_path, f"{lat_c:.3f}_{lon_c:.3f}")
            if not os.path.isdir(sample_path):
                try:
                    os.mkdir(sample_path)
                except FileExistsError:
                    pass
            save_path = os.path.join(sample_path, f"{date_time.strftime('%Y%m%d_%H%M')}.nc")
            queue.put((lat_c, lon_c, date_c, time_c, lat_dim, lon_dim, lat_step, lon_step, save_path))

    for i in range(proc_count):
        queue.put(None)

    proc = []
    for i in range(proc_count):
        p = mp.Process(
            target=GetSample_Worker,
            args=(queue, w_ds)
        )
        p.start()
        proc.append(p)
    for p in proc:
        p.join()

def GetSample_Worker(queue:mp.Queue, w_ds):
    """
    Process the jobs in the queue to get weather samples and save them to disk.

    Args:
        queue (mp.Queue): The job queue.
        w_ds: The weather dataset.

    Returns:
        None
    """

    # child_logger = logger.getChild(f"GetSample_Worker@{os.getppid()}")
    # child_logger.info("Start!")
    while (queue.qsize()):
        try:
            job = queue.get(timeout=5)
        except:
            continue
        if not (type(job) is tuple):
            break
        # child_logger.debug(f"Remaining queue size: {queue.qsize()}")
        lat_c, lon_c, date_c, time_c, lat_dim, lon_dim, lat_step, lon_step, save_path = job
        s_ds = GetSample(w_ds, "", lat_c, lon_c, date_c, time_c, lat_dim, lon_dim, lat_step, lon_step)
        if not s_ds:
            continue
        SaveToDisk(save_path, s_ds)
    # child_logger.info("Exit!")
    exit()
