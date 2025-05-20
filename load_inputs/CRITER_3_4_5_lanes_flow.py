import sys 
import os 
import pandas as pd
import numpy as np
import torch 
from datetime import datetime
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from utils.utilities import restrain_df_to_specific_period
from build_inputs.load_preprocessed_dataset import load_input_and_preprocess

''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - add argument 'num_nodes', 'C' to the NameSpace. These are specific to this data
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''

DATA_SUBFOLDER = 'CRITER_3_4_5_lanes'
NAME = 'flow'
START = '03/01/2019'
END = '06/01/2019'
USELESS_DATES = {'hour':[], #[1,2,3,4,5,6],  #[] if no useless (i.e removed) hours
                 'weekday':[]#[5,6],
                 }
FREQ = '6min'
list_of_invalid_period = []
list_of_invalid_period.append([datetime(2019,3,1,15,30),datetime(2019,3,1,15,42)])
list_of_invalid_period.append([datetime(2019,3,23,7,30),datetime(2019,3,23,9,0)])
list_of_invalid_period.append([datetime(2019,3,23,16,36),datetime(2019,3,23,20,12)])
list_of_invalid_period.append([datetime(2019,3,26,21,0),datetime(2019,3,26,21,12)])
list_of_invalid_period.append([datetime(2019,3,31,2,0),datetime(2019,3,31,3,0)])
C = 1
CITY = 'Lyon'
CHANNEL = 'DEBIT_HEURE'

def load_data(FOLDER_PATH,invalid_dates,coverage_period,args,normalize= True,
              data_subfolder = DATA_SUBFOLDER,
              name=NAME,
              channel = CHANNEL,
              city = CITY,
              ): # args,FOLDER_PATH,coverage_period = None
    assert '6min' == args.freq, f"Trying to apply a a {args.freq} temporal aggregation while CRITER is designed for 6min"

    pivoted_df_full = load_csvs(args,FOLDER_PATH,coverage_period,data_subfolder = data_subfolder,channel = channel,limit_max_nan=200,taux_heure_limit = 100)
    data_T = torch.tensor(pivoted_df_full.values).float()
    dims = [0]# [0]  -> We are normalizing each time-serie independantly 


    processed_input = load_input_and_preprocess(dims = dims,normalize=normalize,invalid_dates=invalid_dates,args=args,data_T=data_T,coverage_period = coverage_period,name=f"{DATA_SUBFOLDER}_{name}") 

    # --- Finalisation Métadonnées ---
    processed_input.spatial_unit = pivoted_df_full.columns.tolist()
    processed_input.C = C
    processed_input.adj_mx_path = f"{FOLDER_PATH}/{data_subfolder}/{data_subfolder}_adj.npy"
    processed_input.distance_mx_path = f"{FOLDER_PATH}/{data_subfolder}/{data_subfolder}_adj.npy"
    processed_input.raw_data_path =f"{FOLDER_PATH}/{data_subfolder}/{data_subfolder}.csv"
    processed_input.city = city
    return processed_input



def load_csvs(args,FOLDER_PATH,coverage_period,data_subfolder,channel,limit_max_nan=200,taux_heure_limit = 100):
    """
    Load the csv file, and return a pivoted dataframe with the time as index and the sensors as columns.
    args:
    -----
    - args : argparse.Namespace contains the entire configuration
    - FOLDER_PATH : path to the folder containing the csv files
    - coverage_period : period to restrain the data to
    - data_subfolder : subfolder containing the csv files
    - channel : channel to load (DEBIT_HEURE or TAUX_HEURE)
    - limit_max_nan : maximum number of nan values allowed in a column. Otherwise, the column is removed.
    - taux_heure_limit : maximum value for the TAUX_HEURE channel (usually consider than > 80 is invalid)

    """

    # Load df: 
    df = pd.read_csv(f"{FOLDER_PATH}/{data_subfolder}/{data_subfolder}.csv",index_col = 0)
    df.HORODATE = pd.to_datetime(df.HORODATE)

    # Mask forbidden values: 
    mask = (df.TAUX_HEURE >= taux_heure_limit) | (df.DEBIT_HEURE < 0)
    df.loc[mask,['TAUX_HEURE','DEBIT_HEURE']] = np.nan

    # Pivot df : 
    pivoted_df = df.pivot_table(index = 'HORODATE',columns = 'ID_POINT_MESURE',values =channel).sort_index()
    reindex = pd.date_range(start = START,end =END,freq = FREQ)[:-1]
    invalid_dates = reindex.difference(pivoted_df.index)
    print(f"Number of invalid time-slots (i.e data when every single sensors does not have data): {len(invalid_dates)}")
    pivoted_df = pivoted_df.reindex(reindex)
    pivoted_df = restrain_df_to_specific_period(pivoted_df,coverage_period)

    # Fill values if expected during nightime (freeflow):
    df_filled = fill_nan_value_when_expected_freeflow(pivoted_df)

    # Fill remaining NaN by interpolation but limited with only one NaN value 
    df_interpolated = df_filled.interpolate(limit=1)

    filtered_df,sparse_columns = remove_sparse_sensor(df_interpolated,limit_max_nan+len(invalid_dates))

    # Resample if needed: 
    if args.freq != '6min':
        pivoted_df = pivoted_df.resample(args.freq).mean()


    print('Number of sensors after filter sparse sensor : ',len(filtered_df.columns))
    print(f" Data loaded with shape: {filtered_df.shape}")
    return filtered_df


def remove_sparse_sensor(df,limit_max_nan = 200):
    # Remove Sesor with too many NaN values: 
    s_nb_nan_per_columns = df.copy().isna().sum()
    sparse_columns = s_nb_nan_per_columns[s_nb_nan_per_columns>limit_max_nan].index
    filtered_df = df.drop(columns = sparse_columns)

    # Input missing data by clusteing:

    print('nb sparse_columns : ',len(sparse_columns))

    return filtered_df,sparse_columns

def fill_nan_value_when_expected_freeflow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace each NaN in the DataFrame (only from 00:00 to 05:00) with the average
    recorded for the same day of the week, the same hour, and the same minute.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame indexed by a DatetimeIndex, with arbitrary columns (sensors).
    
    Returns:
    --------
    pd.DataFrame
        Copy of the DataFrame with NaNs potentially filled for hours 00:00-05:00.
    """
    print('number of nan values before filling : ',df.isna().sum().sum())
    # Mask for night time : from midnight to 6 am  (excluded)
    night_mask = df.index.hour.isin(range(6))

    # Compute grouped mean (weekday, hour, minute):
    means = df.groupby([df.index.weekday, df.index.hour, df.index.minute]).transform('mean')

    # Mask of (time-slot, sensor) to fill : NaN and night time
    mask_night_and_NaN = night_mask[:, None] & df.isna()

    # Fill Values only if NaN within the mask 
    df_filled = df.where(~mask_night_and_NaN,means).copy()

    return df_filled