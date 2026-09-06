import os
import multiprocessing as mp
import pandas as pd
import datetime
import argparse

from libctg_HurricaneTrackDataset import *
from libctg_WeatherDataset import *
import utilities
import configs
from config_loader import CONFIG

def PastDomain(WD: WeatherDataset, HT: HurricaneTrack, htdf:pd.DataFrame, wd_files:list[str], positive_path:str, negative_path:str, nstep:int):
    """Extract positive sample at t=0 and negatives at t-1..t-nstep (past times)."""
    htdf["ISO_TIME"] = pd.to_datetime(htdf["ISO_TIME"])  # ensure datetime

    # for i in htdf.index:
    #     storm_id = htdf["SID"][i]
    #     lat_c = float(htdf["LAT"][i])
    #     lon_c = float(htdf["LON"][i])
    #     current_datetime = htdf["ISO_TIME"][i]

    #     # --- Positive sample at t=0 ---
    #     target = f"{current_datetime.strftime('%Y%m%d')}_{current_datetime.strftime('%H')}_{current_datetime.strftime('%M')}"
    #     matches = [p for p in wd_files if target in os.path.basename(p)]
    #     if len(matches) != 1:
    #         continue
    #     w_ds = WD.LoadFromDisk(matches[0])

    #     lat_nearest = FindNearest(w_ds["latitude"].values, lat_c)
    #     lon_nearest = FindNearest(w_ds["longitude"].values, lon_c)
    #     s_ds = WD.GetSample(
    #         w_ds, storm_id, lat_nearest, lon_nearest,
    #         current_datetime.date(), current_datetime.time(),
    #         lat_dim=WD.DIM_LAT, lon_dim=WD.DIM_LON
    #     )
    #     if not s_ds:
    #         continue

    #     pos_path = os.path.join(output_path, f"POSITIVE_{storm_id}.nc")
    #     WD.SaveToDisk(pos_path, s_ds)

    #     # Use nearest grid point for negatives
    #     lat_c = lat_nearest
    #     lon_c = lon_nearest

    #     # --- Negatives at past time steps ---
    #     for j in range(1, nstep + 1):
    #         sel_dt = current_datetime - datetime.timedelta(hours=WD.STEP_TIME_HOURS * j)
    #         target = f"{sel_dt.strftime('%Y%m%d')}_{sel_dt.strftime('%H')}_{sel_dt.strftime('%M')}"
    #         matches = [p for p in wd_files if target in os.path.basename(p)]
    #         if len(matches) != 1:
    #             break  # stop if no dataset further back

    #         n_w_ds = WD.LoadFromDisk(matches[0])
    #         n_s_ds = WD.GetSample(
    #             n_w_ds, storm_id, lat_c, lon_c,
    #             sel_dt.date(), sel_dt.time(),
    #             lat_dim=WD.DIM_LAT, lon_dim=WD.DIM_LON,
    #             negative_type=f"PAST_T-{j}"
    #         )
    #         if not n_s_ds:
    #             continue

    #         neg_path = os.path.join(
    #             output_path,
    #             f"NEGATIVE_{storm_id}_{j}_{sel_dt.strftime('%Y%m%d_%H%M')}.nc"
    #         )
    #         WD.SaveToDisk(neg_path, n_s_ds)

    # return

    for i in htdf.index:
        # Find for postitive sample
        id = htdf["SID"][i]
        lat_c = htdf["LAT"][i]
        lat_c = float(lat_c)
        lon_c = htdf["LON"][i]
        lon_c = float(lon_c) 
        date_c = htdf["ISO_TIME"][i].date()
        time_c = htdf["ISO_TIME"][i].time()
        target = f"{date_c.strftime('%Y%m%d')}_{time_c.strftime('%H')}_{time_c.strftime('%M')}"
        target_path = [path for path in wd_files if str(os.path.basename(path)).__contains__(target)]
        if len(target_path) != 1:
            continue
        target_path = target_path[0]
        w_ds = WD.LoadFromDisk(target_path)
        lat_c = FindNearest(w_ds["latitude"].values, lat_c)
        lon_c = FindNearest(w_ds["longitude"].values, lon_c)
        s_ds = WD.GetSample(w_ds, id, lat_c, lon_c, date_c, time_c, lat_dim=WD.DIM_LAT, lon_dim=WD.DIM_LON)
        if not s_ds:
            continue
        save_path = os.path.join(positive_path, f"POSITIVE_{id}.nc")
        WD.SaveToDisk(save_path, s_ds)
        current_datetime = htdf["ISO_TIME"][i]
        for j in range(nstep):
            selected_datetime = current_datetime - datetime.timedelta(hours=(WD.STEP_TIME_HOURS*(j+1)))
            date_c = selected_datetime.date()
            time_c = selected_datetime.time()
            target = f"{date_c.strftime('%Y%m%d')}_{time_c.strftime('%H')}_{time_c.strftime('%M')}"
            target_path = [path for path in wd_files if str(os.path.basename(path)).__contains__(target)]
            if len(target_path) != 1:
                continue
            target_path = target_path[0]
            n_w_ds = WD.LoadFromDisk(target_path)
            n_s_ds = WD.GetSample(n_w_ds, id, lat_c, lon_c, date_c, time_c, lat_dim=WD.DIM_LAT, lon_dim=WD.DIM_LON, negative_type=f"PAST_T-{j+1}")
            if not n_s_ds:
                continue
            datetime.datetime.strftime
            save_path = os.path.join(negative_path, f"NEGATIVE_{id}_{j+1}_{selected_datetime.strftime('%Y%m%d_%H%M')}.nc")
            WD.SaveToDisk(save_path, n_s_ds)
            pass
        pass
    return

def Worker(queue:mp.Queue, WD:WeatherDataset, HT:HurricaneTrack, wd_files:list[str], positive_path:str, negative_path, nstep:int):
    while (queue.qsize()):
        htdf = queue.get()
        if (not type(htdf) == pd.DataFrame):
            continue
        PastDomain(WD, HT, htdf, wd_files, positive_path, negative_path ,nstep)
    print("Done.")
    exit()

def PastDomain_Main(WD: WeatherDataset, HT: HurricaneTrack, PrepDir:str, PositiveOutDir:str, NegativeOutDir:str, nworker:int=1):
    ht_files = utilities.RecurseListDir(CONFIG.OPATH.TRACKS_PREP, ["FIRST_*.csv"])
    wd_files = utilities.RecurseListDir(PrepDir, ["*.nc"])
    htdf = HT.LoadBatch(ht_files)
    htdfs = HT.Split(htdf, CONFIG.DOMAIN_EXTRACTION_BATCH_SIZE)
    queue = mp.Queue()
    for h in htdfs:
        queue.put(h)
    utilities.MultiProcessing(Worker, (queue, WD, HT, wd_files, PositiveOutDir, NegativeOutDir, CONFIG.STEP_BACK_COUNT), nworker)
    return

def Fnl():
    print("NCEP-FNL dataset extraction")
    FNL = NcepFnl()
    IBTRACS = Ibtracs()
    IBTRACS.TIME_POINTS_HOURS = [0, 6, 12, 18]
    # Prepare IBTRACS
    df_first_path = os.path.join(CONFIG.OPATH.TRACKS_PREP, "FIRST_FNL_IBTRACS.csv")
    utilities.CleanDir(CONFIG.OPATH.TRACKS_PREP)
    files = utilities.RecurseListDir(CONFIG.IPATH.TRACKS_RAW, ["*.csv"])
    df_raw = IBTRACS.LoadRawCSVs(files)
    df_first = IBTRACS.ProcessRaw(df_raw)
    IBTRACS.SaveToDisk(df_first_path, df_first)
    # Past domain extraction
    positive_out_dir = CONFIG.OPATH.NCEP_POSITIVE
    utilities.CleanDir(positive_out_dir)
    negative_out_dir = CONFIG.OPATH.NCEP_PAST
    utilities.CleanDir(negative_out_dir)
    PastDomain_Main(FNL, IBTRACS, CONFIG.OPATH.NCEP_PREP, positive_out_dir, negative_out_dir, configs.nworkers.FNL_PASTDOMAIN)   
    print("NCEP-FNL dataset extraction: Done.")
    return

def Merra2():
    print("NASA-MERRA2 dataset extraction")
    MERRA2 = NasaMerra2()
    IBTRACS = Ibtracs()
    IBTRACS.TIME_POINTS_HOURS = [0,3,6,9,12,15,18,21]
    # Prepare IBTRACS
    df_first_path = os.path.join(CONFIG.OPATH.TRACKS_PREP, "FIRST_MERRA2_IBTRACS.csv")
    utilities.CleanDir(CONFIG.OPATH.TRACKS_PREP)
    files = utilities.RecurseListDir(CONFIG.IPATH.TRACKS_RAW, ["*.csv"])
    df_raw = IBTRACS.LoadRawCSVs(files)
    df_first = IBTRACS.ProcessRaw(df_raw)
    IBTRACS.SaveToDisk(df_first_path, df_first)
    # Past domain extraction
    positive_out_dir = CONFIG.OPATH.MERRA2_POSITIVE
    utilities.CleanDir(positive_out_dir)
    negative_out_dir = CONFIG.OPATH.MERRA2_PAST
    utilities.CleanDir(negative_out_dir)
    PastDomain_Main(MERRA2, IBTRACS, CONFIG.OPATH.MERRA2_PREP, positive_out_dir, negative_out_dir, configs.nworkers.MERRA2_PASTDOMAIN)   
    print("NASA-MERRA2 dataset extraction: Done.")
    return

def Cmip6():
    print("CMIP6 dataset extraction")
    CMIP6 = WrfCmip6()
    CMIP6TRACKS = Cmip6Tracks()
    # Prepare CMIP6TRACKS
    df_first_path = os.path.join(CONFIG.OPATH.TRACKS_PREP, "FIRST_CMIP6_CMIP6TRACS.csv")
    utilities.CleanDir(CONFIG.OPATH.TRACKS_PREP)
    files = utilities.RecurseListDir(CONFIG.IPATH.TRACKS_RAW, ["*.txt"])
    df_raw = CMIP6TRACKS.LoadRawTxts(files)
    df_first = CMIP6TRACKS.ProcessRaw(df_raw)
    CMIP6TRACKS.SaveToDisk(df_first_path, df_first)
    # Past domain extraction
    positive_out_dir = CONFIG.OPATH.CMIP6_POSITIVE
    utilities.CleanDir(positive_out_dir)
    negative_out_dir = CONFIG.OPATH.CMIP6_PAST
    utilities.CleanDir(negative_out_dir)
    PastDomain_Main(CMIP6, CMIP6TRACKS, CONFIG.OPATH.CMIP6_PREP, positive_out_dir, negative_out_dir, configs.nworkers.MERRA2_PASTDOMAIN)   
    print("CMIP6 dataset extraction: Done.")
    return

def Era5_extract():
    print("ERA5 dataset extraction")
    ERA5 = Era5()
    IBTRACS = Ibtracs()
    IBTRACS.TIME_POINTS_HOURS = [0, 6, 12, 18]
    # Prepare IBTRACS
    df_first_path = os.path.join(CONFIG.OPATH.TRACKS_PREP, "FIRST_ERA5_IBTRACS.csv")
    utilities.CleanDir(CONFIG.OPATH.TRACKS_PREP)
    files = utilities.RecurseListDir(CONFIG.IPATH.TRACKS_RAW, ["*.csv"])
    df_raw = IBTRACS.LoadRawCSVs(files)
    df_first = IBTRACS.ProcessRaw(df_raw)
    print(df_first_path)
    print(df_first)
    result = IBTRACS.SaveToDisk(df_first_path, df_first)
    print(result)
    # Past domain extraction
    positive_out_dir = CONFIG.OPATH.ERA5_POSITIVE
    utilities.CleanDir(positive_out_dir)
    negative_out_dir = CONFIG.OPATH.ERA5_PAST
    utilities.CleanDir(negative_out_dir)
    PastDomain_Main(ERA5, IBTRACS, CONFIG.OPATH.ERA5_PREP, positive_out_dir, negative_out_dir, configs.nworkers.MERRA2_PASTDOMAIN)   
    print("ERA5 dataset extraction: Done.")
    return

def Gfs_Extract():
    print("GFS dataset extraction")
    GFS = Gfs()
    IBTRACS = Ibtracs()
    IBTRACS.TIME_POINTS_HOURS = [0, 6, 12, 18]
    # Prepare IBTRACS
    df_first_path = os.path.join(CONFIG.OPATH.TRACKS_PREP, "FIRST_GFS_IBTRACS.csv")
    utilities.CleanDir(CONFIG.OPATH.TRACKS_PREP)
    files = utilities.RecurseListDir(CONFIG.IPATH.TRACKS_RAW, ["*.csv"])
    df_raw = IBTRACS.LoadRawCSVs(files)
    df_first = IBTRACS.ProcessRaw(df_raw)
    print(df_first_path)
    print(df_first)
    result = IBTRACS.SaveToDisk(df_first_path, df_first)
    print(result)
    # Past domain extraction
    positive_out_dir = CONFIG.OPATH.GFS_POSITIVE
    utilities.CleanDir(positive_out_dir)
    negative_out_dir = CONFIG.OPATH.GFS_PAST
    utilities.CleanDir(negative_out_dir)
    PastDomain_Main(GFS, IBTRACS, CONFIG.OPATH.GFS_PREP, positive_out_dir, negative_out_dir, configs.nworkers.MERRA2_PASTDOMAIN)   
    print("GFS dataset extraction: Done.")
    return

def Main():
    print("Past Domain")
    parser = argparse.ArgumentParser(description="Extract Past Domain Samples")
    parser.add_argument('--dataset', type=str, required=True, choices=['fnl', 'merra2', 'cmip6', 'era5', 'gfs'], help='Dataset to process')
    args = parser.parse_args()
    match args.dataset.lower():
        case 'fnl':
            Fnl()
        case 'merra2':
            Merra2()
        case 'cmip6':
            Cmip6()
        case 'era5':
            Era5_extract()
        case 'gfs':
            Gfs_Extract()
        case _:
            raise ValueError(f"Unknown dataset: {args.dataset}")
    print("Past Domain: Done.")
    return

if __name__=="__main__":
    utilities.GetNumberOfWorker()
    Main()
