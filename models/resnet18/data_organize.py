import os

import pandas as pd

# from tqdm import tqdm

#################
# Table columns:
# - Path: str
# - Filename: str
# - ID: str
# - Year: int
# - Domain: str
# - Position: int
# - Step: int
# - Noise: bool
#################

# 20240305
# 20240424
# 20240806
# 20240809

def organize():

    #domain_dir = '/N/u/tqluu/BigRed200/@PUBLIC/20240814/nasa-merra2'
    #data_csv = '/N/slate/ckieu/typhoon-formation/output/csv/ERA5_1981_1985.csv'
    domain_dir = '/N/slate/ckieu/tcg-net/output/'
    out_dir = '/N/slate/ckieu/tcg-net/output/csv'
    os.makedirs(out_dir, exist_ok=True)
    data_csv = f'{out_dir}/all.csv'

    df = []

    # process positive
    dir_path = os.path.join(domain_dir, 'POSITIVE')
    list_file = [(path, file)
                for path, _, files in os.walk(dir_path)
                for file in files]
    list_file.sort()
    print('POSITIVE:', len(list_file))
    for path, file in list_file:
        #print(file,file.split('_')[-1].split('.')[0])
        Path = path + '/' + file
        ID = file.split('_')[-1].split('.')[0]
        Year = int(ID[: 4])
        Domain = 'POSITIVE'
        Position = 0
        Step = 0
        
        row = [Path, file, ID, Year, Domain, Position, Step]
        df.append(row)
        
    # process past domain
    dir_path = os.path.join(domain_dir, 'PastDomain')
    list_file = [(path, file)
                for path, _, files in os.walk(dir_path)
                for file in files]
    list_file.sort()
    print('PastDomain:', len(list_file))
    for path, file in list_file:
        Path = path + '/' + file
        ID = file.split('_')[1]
        Year = int(ID[: 4])
        Domain = 'Past'
        Position = 0
        Step = int(file.split('_')[2].split('.')[0])
        
        row = [Path, file, ID, Year, Domain, Position, Step]
        df.append(row)
        
    # process dynamic domain (new from step 1 to 40)
    dir_path = os.path.join(domain_dir, 'DynamicDomain')
    list_file = [(path, file)
                for path, _, files in os.walk(dir_path)
                for file in files]
    list_file.sort()
    print('DynamicDomain2:', len(list_file))
    direction = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w']

    for path, file in list_file:
        Path = path + '/' + file
        ID = file.split('_')[1]
        Year = int(ID[: 4])
        Domain = 'Dynamic'
        Position = direction.index(file.split('_')[2]) + 1
        Step = int(file.split('_')[3].split('.')[0])
        
        row = [Path, file, ID, Year, Domain, Position, Step]
        df.append(row)
        
    df = pd.DataFrame(df, columns=['Path', 'FileName', 'ID', 'Year', 'Domain', 'Position', 'Step'])
    df.to_csv(data_csv, index=None)

if __name__ == '__main__':
    organize()
