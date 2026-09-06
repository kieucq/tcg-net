import os
import multiprocessing as mp
import pandas as pd
import xarray as xr
import datetime
import argparse

from libctg_HurricaneTrackDataset import *
from libctg_WeatherDataset import *
import utilities
import configs
from config_loader import CONFIG

DOMAIN_LOCATIONS = ["n","ne","e","se","s","sw","w","nw"]

def DynamicDomain(WD: WeatherDataset, HT: HurricaneTrack, htdf:pd.DataFrame, wd_files:list[str], positive_path:str, negative_path:str, ndomains:list[str], n_steps:int):
    htdf["ISO_TIME"] = pd.to_datetime(htdf["ISO_TIME"])  # Convert to datetime
    for i in htdf.index:
        # --- Storm info ---
        id = htdf["SID"][i]
        lat_c = float(htdf["LAT"][i])
        lon_c = float(htdf["LON"][i])
        current_datetime = htdf["ISO_TIME"][i]

        # --- t=0 dataset lookup ---
        date_c = current_datetime.date()
        time_c = current_datetime.time()
        target = f"{date_c.strftime('%Y%m%d')}_{time_c.strftime('%H')}_{time_c.strftime('%M')}"
        target_path = [
            path for path in wd_files if str(os.path.basename(path)).__contains__(target)
        ]

        if len(target_path) != 1:
            # No dataset for this row → skip to next record
            continue

        # --- Load dataset for t=0 ---
        w_ds = WD.LoadFromDisk(target_path[0])

        # --- Positive sample (only at t=0) ---
        lat_nearest = FindNearest(w_ds["latitude"].values, lat_c)
        lon_nearest = FindNearest(w_ds["longitude"].values, lon_c)
        s_ds = WD.GetSample(
            w_ds, id, lat_nearest, lon_nearest, date_c, time_c,
            lat_dim=WD.DIM_LAT, lon_dim=WD.DIM_LON
        )
        if not s_ds:
            continue

        save_path = os.path.join(positive_path, f"POSITIVE_{id}.nc")
        WD.SaveToDisk(save_path, s_ds)

        # reset center for negatives
        lat_c = findMiddle(s_ds["latitude"].values.tolist())
        lon_c = findMiddle(s_ds["longitude"].values.tolist())

        # --- Negatives for t=0..n_steps ---
        for t in range(0, n_steps + 1):
            sel_datetime = current_datetime - datetime.timedelta(hours=WD.STEP_TIME_HOURS * t)
            date_c = sel_datetime.date()
            time_c = sel_datetime.time()
            target = f"{date_c.strftime('%Y%m%d')}_{time_c.strftime('%H')}_{time_c.strftime('%M')}"
            target_path = [
                path for path in wd_files if str(os.path.basename(path)).__contains__(target)
            ]
            if len(target_path) != 1:
                # no dataset → stop looking further back in time
                break

            w_ds = WD.LoadFromDisk(target_path[0])

            for domain in ndomains:
                # domain shift
                match domain:
                    case "n":
                        n_lat_c = lat_c + int(WD.DIM_LAT) * WD.STEP_LAT
                        n_lon_c = lon_c
                    case "ne":
                        n_lat_c = lat_c + int(WD.DIM_LAT) * WD.STEP_LAT
                        n_lon_c = lon_c + int(WD.DIM_LON) * WD.STEP_LON
                    case "e":
                        n_lat_c = lat_c
                        n_lon_c = lon_c + int(WD.DIM_LON) * WD.STEP_LON
                    case "se":
                        n_lat_c = lat_c - int(WD.DIM_LAT) * WD.STEP_LAT
                        n_lon_c = lon_c + int(WD.DIM_LON) * WD.STEP_LON
                    case "s":
                        n_lat_c = lat_c - int(WD.DIM_LAT) * WD.STEP_LAT
                        n_lon_c = lon_c
                    case "sw":
                        n_lat_c = lat_c - int(WD.DIM_LAT) * WD.STEP_LAT
                        n_lon_c = lon_c - int(WD.DIM_LON) * WD.STEP_LON
                    case "w":
                        n_lat_c = lat_c
                        n_lon_c = lon_c - int(WD.DIM_LON) * WD.STEP_LON
                    case "nw":
                        n_lat_c = lat_c + int(WD.DIM_LAT) * WD.STEP_LAT
                        n_lon_c = lon_c - int(WD.DIM_LON) * WD.STEP_LON
                    case _:
                        raise ValueError(f"{domain} is invalid!!!")

                n_s_ds = WD.GetSample(
                    w_ds, id, n_lat_c, n_lon_c, date_c, time_c,
                    lat_dim=WD.DIM_LAT, lon_dim=WD.DIM_LON,
                    negative_type=f"DYNAMIC_{domain}_{t}"
                )
                if n_s_ds:
                    save_path = os.path.join(negative_path, f"NEGATIVE_{id}_{domain}_{t}.nc")
                    WD.SaveToDisk(save_path, n_s_ds)
    return

    # htdf["ISO_TIME"] = pd.to_datetime(htdf["ISO_TIME"]) # Convert to datetime
    # for i in htdf.index:
    #     # Find for postitive sample
    #     id = htdf["SID"][i]
    #     lat_c = htdf["LAT"][i]
    #     lat_c = float(lat_c)
    #     lon_c = htdf["LON"][i]
    #     lon_c = float(lon_c) 
    #     date_c = htdf["ISO_TIME"][i].date()
    #     time_c = htdf["ISO_TIME"][i].time()
    #     target = f"{date_c.strftime('%Y%m%d')}_{time_c.strftime('%H')}_{time_c.strftime('%M')}"
    #     target_path = [path for path in wd_files if str(os.path.basename(path)).__contains__(target)]
    #     if len(target_path) != 1:
    #         continue
    #     target_path = target_path[0]
    #     w_ds = WD.LoadFromDisk(target_path)
    #     lat_c = FindNearest(w_ds["latitude"].values, lat_c)
    #     lon_c = FindNearest(w_ds["longitude"].values, lon_c)
    #     s_ds = WD.GetSample(w_ds, id, lat_c, lon_c, date_c, time_c, lat_dim=WD.DIM_LAT, lon_dim=WD.DIM_LON)
    #     if not s_ds:
    #         continue
    #     save_path = os.path.join(output_path, f"POSITIVE_{id}.nc")
    #     WD.SaveToDisk(save_path, s_ds)
    #     lat_c = findMiddle(s_ds["latitude"].values.tolist())
    #     lon_c = findMiddle(s_ds["longitude"].values.tolist())
    #     current_datetime = htdf["ISO_TIME"][i]
    #     t = 0
    #     while True:
    #         # Negative extraction
    #         for domain in ndomains:
    #             match domain:
    #                 case "n":
    #                     n_lat_c = lat_c + int(WD.DIM_LAT)*WD.STEP_LAT
    #                     n_lon_c = lon_c + 0
    #                 case "ne":
    #                     n_lat_c = lat_c + int(WD.DIM_LAT)*WD.STEP_LAT
    #                     n_lon_c = lon_c + int(WD.DIM_LON)*WD.STEP_LON
    #                 case "e":
    #                     n_lat_c = lat_c + 0
    #                     n_lon_c = lon_c + int(WD.DIM_LON)*WD.STEP_LON
    #                 case "se":
    #                     n_lat_c = lat_c - int(WD.DIM_LAT)*WD.STEP_LAT
    #                     n_lon_c = lon_c + int(WD.DIM_LON)*WD.STEP_LON
    #                 case "s":
    #                     n_lat_c = lat_c - int(WD.DIM_LAT)*WD.STEP_LAT
    #                     n_lon_c = lon_c + 0
    #                 case "sw":
    #                     n_lat_c = lat_c - int(WD.DIM_LAT)*WD.STEP_LAT
    #                     n_lon_c = lon_c - int(WD.DIM_LON)*WD.STEP_LON
    #                 case "w":
    #                     n_lat_c = lat_c - 0
    #                     n_lon_c = lon_c - int(WD.DIM_LON)*WD.STEP_LON
    #                 case "nw":
    #                     n_lat_c = lat_c + int(WD.DIM_LAT)*WD.STEP_LAT
    #                     n_lon_c = lon_c - int(WD.DIM_LON)*WD.STEP_LON
    #                 case _:
    #                     raise f"{domain} is invalid!!!"
    #             n_s_ds = WD.GetSample(w_ds, id, n_lat_c, n_lon_c, date_c, time_c, lat_dim=WD.DIM_LAT, lon_dim=WD.DIM_LON, negative_type=f"DYNAMIC_{domain}_{t}")
    #             if not n_s_ds:
    #                 continue
    #             save_path = os.path.join(output_path, f"NEGATIVE_{id}_{domain}_{t}.nc")
    #             WD.SaveToDisk(save_path, n_s_ds)
    #             pass
    #         # Return to the past
    #         t += 1
    #         if t > n_steps:
    #             break
    #         selected_datetime = current_datetime - datetime.timedelta(hours=(WD.STEP_TIME_HOURS*(t)))
    #         date_c = selected_datetime.date()
    #         time_c = selected_datetime.time()
    #         target = f"{date_c.strftime('%Y%m%d')}_{time_c.strftime('%H')}_{time_c.strftime('%M')}"
    #         target_path = [path for path in wd_files if str(os.path.basename(path)).__contains__(target)]
    #         print(target)
    #         print(target_path)
    #         if len(target_path) != 1:
    #             continue
    #         target_path = target_path[0]
    #         print(f"past: {target_path}")
    #         w_ds = WD.LoadFromDisk(target_path)
    #         pass
    #     pass
    # return

def Worker(queue:mp.Queue, WD:WeatherDataset, HT:HurricaneTrack, wd_files:list[str], positive_path:str, negative_path:str, ndomains:list[str], nsteps:int):
    while (queue.qsize()):
        htdf = queue.get()
        if (not type(htdf) == pd.DataFrame):
            continue
        DynamicDomain(WD, HT, htdf, wd_files, positive_path, negative_path, ndomains, nsteps)
        pass
    print("Done.")
    exit()

def DynamicDomain_Main(WD: WeatherDataset, HT: HurricaneTrack, PrepDir:str, PositiveOutDir:str, NegativeOutDir:str, nworker:int=1):
    ht_files = utilities.RecurseListDir(CONFIG.OPATH.TRACKS_PREP, ["FIRST_*.csv"])
    print(ht_files)
    wd_files = utilities.RecurseListDir(PrepDir, ["*.nc"])
    htdf = HT.LoadBatch(ht_files)
    htdfs = HT.Split(htdf, CONFIG.DOMAIN_EXTRACTION_BATCH_SIZE)
    queue = mp.Queue()
    for h in htdfs:
        queue.put(h)
        pass
    # mp.Process(target=utilities.ProgressBar, args=(queue, len(htdfs), "Dynamic Domain Extraction")).start()
    utilities.MultiProcessing(Worker, (queue, WD, HT, wd_files, PositiveOutDir, NegativeOutDir, DOMAIN_LOCATIONS, CONFIG.STEP_BACK_COUNT), nworker)
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
    # Dynamic Domain Extraction
    positive_out_dir = CONFIG.OPATH.NCEP_POSITIVE
    utilities.CleanDir(positive_out_dir)
    negative_out_dir = CONFIG.OPATH.NCEP_DYNAMIC
    utilities.CleanDir(negative_out_dir)
    DynamicDomain_Main(FNL, IBTRACS, CONFIG.OPATH.NCEP_PREP, positive_out_dir, negative_out_dir ,configs.nworkers.FNL_DYNAMICDOMAIN)  
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
    # Dynamic Domain Extraction
    positive_out_dir = CONFIG.OPATH.MERRA2_POSITIVE
    utilities.CleanDir(positive_out_dir)
    negative_out_dir = CONFIG.OPATH.MERRA2_DYNAMIC
    utilities.CleanDir(negative_out_dir)
    DynamicDomain_Main(MERRA2, IBTRACS, CONFIG.OPATH.MERRA2_PREP, positive_out_dir, negative_out_dir, configs.nworkers.MERRA2_DYNAMICDOMAIN)      
    print("NASA-MERRA2 dataset extraction: Done")
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
    # Dynamic Domain Extraction
    positive_out_dir = CONFIG.OPATH.CMIP6_POSITIVE
    utilities.CleanDir(positive_out_dir)
    negative_out_dir = CONFIG.OPATH.CMIP6_DYNAMIC
    utilities.CleanDir(negative_out_dir)
    DynamicDomain_Main(CMIP6, CMIP6TRACKS, CONFIG.OPATH.CMIP6_PREP, positive_out_dir, negative_out_dir, configs.nworkers.MERRA2_DYNAMICDOMAIN)      
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
    IBTRACS.SaveToDisk(df_first_path, df_first)
    # Dynamic Domain Extraction
    positive_out_dir = CONFIG.OPATH.ERA5_POSITIVE
    utilities.CleanDir(positive_out_dir)
    negative_out_dir = CONFIG.OPATH.ERA5_DYNAMIC
    utilities.CleanDir(negative_out_dir)
    DynamicDomain_Main(ERA5, IBTRACS, CONFIG.OPATH.ERA5_PREP, positive_out_dir, negative_out_dir, 2)      
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
    IBTRACS.SaveToDisk(df_first_path, df_first)
    # Dynamic Domain Extraction
    positive_out_dir = CONFIG.OPATH.GFS_POSITIVE
    utilities.CleanDir(positive_out_dir)
    negative_out_dir = CONFIG.OPATH.GFS_DYNAMIC
    utilities.CleanDir(negative_out_dir)
    DynamicDomain_Main(GFS, IBTRACS, CONFIG.OPATH.GFS_PREP, positive_out_dir, negative_out_dir, 2)      
    print("GFS dataset extraction: Done.")
    return

def Main():
    print("Dynamic Domain")
    parser = argparse.ArgumentParser(description="Extract Dynamic Domain Samples")
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
    print("Dynamic Domain: Done.")
    return

if __name__=="__main__":
    Main()
