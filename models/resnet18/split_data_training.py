import os
import pandas as pd
from math import nan
from pathlib import Path
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

def data_label(args):

    rus_ratio = args.rus_ratio
    data_csv = f'{args.inp_dir}/all.csv'
    out_dir_root = args.out_dir
    train_start_year = args.train_start_year
    train_end_year = args.train_end_year
    test_start_year = args.test_start_year
    test_end_year = args.test_end_year

    dataset = {
        'train': [train_start_year,train_end_year],
        'test': [test_start_year,test_end_year],
    }

    df = pd.read_csv(data_csv)

    for step in range(2, 20, 2):
        out_dir = os.path.join(out_dir_root, f'DynamicRemain_rus{rus_ratio}', f'Step_{step}')
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        print(out_dir)
        
        # prepare train/val
        for ds, irange in dataset.items():
            df['Label'] = nan
            df.loc[
                (df['Year'].between(*irange, inclusive='both')) &
                (df['Position'] == 0) &
                (df['Step']).between(step, step, inclusive='both')
            , 'Label'] = 1
            
            df.loc[
                (df['Year'].between(*irange, inclusive='both')) &
                (df['Position'] == 0) &
                (df['Step']).between(step, 40, inclusive='right')
            , 'Label'] = 0
            
            df.loc[
                (df['Year'].between(*irange, inclusive='both')) &
                (df['Position'] > 0) &
                (df['Step']).between(step, 40, inclusive='both')
            , 'Label'] = 0
            
            if ds == 'train':
                df.loc[df[df['Label'] == 0].sample(max((df['Label'] == 0).sum() - (df['Label'] == 1).sum() * rus_ratio, 0)).index, 'Label'] = nan
                
                try:
                    train, val = train_test_split(df[~ df['Label'].isna()], stratify=df[~ df['Label'].isna()]['Label'], test_size=0.1)
                except:
                    train = val = pd.DataFrame()

                train_df = df.copy()
                train_df.loc[val.index, 'Label'] = nan
                csv_path = os.path.join(out_dir, 'train.csv')
                train_df.to_csv(csv_path, index=False)
                print('train', len(train_df[train_df['Label'] == 1]), len(train_df[train_df['Label'] == 0]), len(train_df[train_df['Label'].isna()]))
                
                val_df = df.copy()
                val_df.loc[train.index, 'Label'] = nan
                csv_path = os.path.join(out_dir, 'val.csv')
                val_df.to_csv(csv_path, index=False)
                print('val', len(val_df[val_df['Label'] == 1]), len(val_df[val_df['Label'] == 0]), len(val_df[val_df['Label'].isna()]))
            
            else:
                csv_path = os.path.join(out_dir, ds + '.csv')
                df.to_csv(csv_path, index=False)
                print('test', len(df[df['Label'] == 1]), len(df[df['Label'] == 0]), len(df[df['Label'].isna()]))
                
                df.loc[df[df['Label'] == 0].sample(max((df['Label'] == 0).sum() - (df['Label'] == 1).sum() * rus_ratio, 0)).index, 'Label'] = nan
                csv_path = os.path.join(out_dir, ds + '2.csv')
                df.to_csv(csv_path, index=False)
                print('test2', len(df[df['Label'] == 1]), len(df[df['Label'] == 0]), len(df[df['Label'].isna()]))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--inp_dir", type=str, default = '../output/csv/')
    parser.add_argument("--out_dir", type=str, default = '../output/csv/')
    parser.add_argument("--train_start_year", type=int, default = 1980)
    parser.add_argument("--train_end_year", type=int, default = 2016)
    parser.add_argument("--test_start_year", type=int, default = 2017)
    parser.add_argument("--test_end_year", type=int, default = 2022)
    parser.add_argument("--rus_ratio", type=int, default = 30)

    args = parser.parse_args()
    data_label(args)
