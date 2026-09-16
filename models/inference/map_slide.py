import datetime
import os, shutil
import pandas as pd
import argparse

from libtcg_SliceWindow import SliceWindow
from utilities.dir import CleanDir
from config_loader import CONFIG

def generateCsv(slice_windows_path, run_opt, agg_steps, time_to_run, save_path):
    points = os.listdir(slice_windows_path)
    rsl = []
    for point in points:
        if run_opt == 0:
            for time_str in time_to_run:
                current = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                data_path = getFilePathFromTime(slice_windows_path + '/' + point, current)
                if os.path.exists(data_path):
                    rsl.append([current, point, data_path, 0])
        elif run_opt == 1:          
            # 3 hours for MERRA2, 6 hours for ERA5, NCEP, CMIP6
            time_step_size = CONFIG.SLICING_WINDOW.AGGREGATION.TIME_STEP_HOURS

            for time_str in time_to_run:
                date_time = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                for step in range(agg_steps):
                    current = date_time - datetime.timedelta(hours=step * time_step_size)
                    data_path = getFilePathFromTime(slice_windows_path + '/' + point, current)
                    if os.path.exists(data_path):
                        rsl.append([date_time, point, data_path, step * -1])

    df = pd.DataFrame(rsl, columns = ['Datetime', 'Point', 'Path', 'Step'])
    
    # trong trường hợp aggregate, kiểm tra và loại bỏ những datetime không có đủ n step trước đó để aggregate
    if run_opt == 1:
        groups = df.groupby(['Datetime', 'Point']).count()
        indexes = groups[groups['Step'] < agg_steps].index
        df = df[~df.set_index(['datetime', 'point']).index.isin(indexes)]

    df['Label'] = 1
    df.to_csv(save_path, index=False)

def getFilePathFromTime(folder_path, date_time):
    year = str(date_time.year).zfill(4)
    month = str(date_time.month).zfill(2)
    day = str(date_time.day).zfill(2)
    hour = str(date_time.hour).zfill(2)
    data_path = folder_path + '/' + f'{year}{month}{day}_{hour}00.nc'
    return data_path

# có thể sửa thêm code để ghi thêm ra log
def slide(data_type='merra2'):    
    # Chuẩn bị các param cho sliding window
    # SliceWindow thực hiện trượt từng ô lưới (pixel) trên ảnh input
    # nếu ô lưới đó nằm trong phạm vi lat_min,lat_max,lon_min,lon_max
    # thì sẽ lấy ô đó làm trung tâm, cắt 1 ảnh con có kích thước [lat_dim, lon_dim] (VD: 17x17), lưu vào thư mục tạm
    
    input_path = CONFIG.SLICING_WINDOW.INPUT_PATH
    output_folder = CONFIG.SLICING_WINDOW.OUTPUT_PATH
    # input_path = '/N/slate/ckieu/typhoon-formation/output/ERA5_extend_test'
    # output_folder = '/N/slate/ckieu/typhoon-formation/output/ERA5_slide_test'

    # Các tham số liên quan đến sliding windows
    # [sliding_window]
    # grid extent: tọa độ 4 góc của lưới output mong muốn
    lat_min = CONFIG.SLICING_WINDOW.AREA.MIN_LAT
    lat_max = CONFIG.SLICING_WINDOW.AREA.MAX_LAT
    lon_min = CONFIG.SLICING_WINDOW.AREA.MIN_LON
    lon_max = CONFIG.SLICING_WINDOW.AREA.MAX_LON
    # LAT_DIM, LON_DIM: kích thước 2 chiều 1 ảnh con sẽ cắt từ ảnh input khi trượt cửa sổ. VD: 17x17
    lat_dim = float(CONFIG.SLICING_WINDOW.CHILD_AREA.LAT_DIM)
    lon_dim = float(CONFIG.SLICING_WINDOW.CHILD_AREA.LON_DIM)
    # STEP_LAT, STEP_LON: độ phân giải của ảnh con. VD: 0.5 độ
    lat_step = float(CONFIG.SLICING_WINDOW.CHILD_AREA.RESOLUTION.LAT_STEP)
    lon_step = float(CONFIG.SLICING_WINDOW.CHILD_AREA.RESOLUTION.LON_STEP)
    # Số bước nhảy khi trượt cửa sổ
    lat_nstep = CONFIG.SLICING_WINDOW.NUM_STEP.LAT_NSTEP
    lon_nstep = CONFIG.SLICING_WINDOW.NUM_STEP.LON_NSTEP
    # thời gian của các file input, có thể sửa code để đọc từ 1 file csv, thay vì khai báo mảng như này
    print(input_path)
    time_to_run = [file for _, _, files in os.walk(input_path) 
               for file in files if file.endswith('.nc')]
    print(time_to_run)
    time_to_run = [datetime.datetime.strptime(file, f'{data_type}_%Y%m%d_%H_00.nc') for file in time_to_run]
    time_to_run = [file.strftime('%Y-%m-%d %H:%M:%S') for file in time_to_run]
    print(len(time_to_run), 'files to run')

    CleanDir(output_folder)
    
    proc_count = 8
    subproc_count = 8

    print('Start sliding window')
    SliceWindow(
        data_type, input_path, output_folder,
        lat_min,lat_max,lon_min,lon_max,
        float(lat_dim),
        float(lon_dim),
        float(lat_step),
        float(lon_step),
        lat_nstep,
        lon_nstep,
        proc_count,
        subproc_count,
        time_to_run,
    )

    #######################
    # dựa trên các ảnh tạo được trong thư mục slice window ->
    # tạo file csv chứa đường dẫn đến các ảnh con, mục đích để sử dụng cho dataloader
    # file csv gồm các trường: datetime point   path    step
    # datetime: thời gian của các file input, lấy từ tham số [time_to_run]
    # point: (điểm), tọa độ của điểm trung tâm ảnh con, dạng [lat_lon]
    # path: đường dẫn đến ảnh trong thư mục slice window
    # step: trong trường hợp không aggregate, step = 0
    # trong trường hợp có aggregate, cùng 1 datetime input sẽ có nhiều step khác nhau: 0, -1, -2, ...
    run_opt = CONFIG.SLICING_WINDOW.AGGREGATION.RUN_OPT
    agg_steps = CONFIG.SLICING_WINDOW.AGGREGATION.AGG_STEPS
    csv_path = output_folder + '/data.csv'
    
    generateCsv(
        output_folder,
        run_opt,
        agg_steps,
        time_to_run,
        csv_path
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map sliding window")
    parser.add_argument('--dataset', type=str, required=True, choices=['fnl', 'merra2', 'cmip6', 'era5', 'gfs'], help='Dataset to process')
    args = parser.parse_args()
    data_type = args.dataset.lower()
    print(f'Processing dataset: {data_type}')
    slide(data_type)
    print('Finished sliding window')
