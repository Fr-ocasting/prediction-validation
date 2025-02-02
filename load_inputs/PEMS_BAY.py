import sys 
import os 
import pandas as pd
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from dataset import DataSet
from datetime import datetime
from utils.utilities import filter_args,get_time_step_per_hour

import numpy as np 
import h5py

''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - add argument 'n_vertex', 'C' to the NameSpace. These are specific to this data
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''

FILE_NAME = 'PEMS_BAY/PEMS_BAY'#'subway_IN_interpol_neg_15_min_2019_2020' #.csv
START = '01/01/2017'
END = '07/01/2017'
FREQ = '5min'

C = 1
n_vertex = 325



list_of_invalid_period = []
# Daylight saving time changeover
list_of_invalid_period.append([datetime(2017,3,12,2,0),datetime(2017,3,12,2,55)])


def load_data(args,ROOT,FOLDER_PATH,coverage_period = None):
    '''Load the dataset. Supposed to coontains pd.DateTime Index as index, and named columns.
    columns has to represent the spatial units.

    outputs: 
    ---------
    df: contains 
    df.index : coverage period of the dataset 
    invalid_dates : list of invalid dates 
    '''
    data = h5py.File(f"{ROOT}/{FOLDER_PATH}/{FILE_NAME}.h5", 'r')
    axis0 = pd.Series(data['speed']['axis0'][:].astype(str))
    axis1 = pd.Series(data['speed']['axis1'][:].astype(str))
    df = pd.DataFrame(data['speed']['block0_values'][:], columns=axis0, index = pd.to_datetime(axis1.astype(int)/1_000_000_000,unit='s'))
    df.columns.name = 'Sensor'
    df.index = pd.to_datetime(df.index)

    

    if args.freq != FREQ :
        assert int(args.freq.replace('min',''))> int(FREQ.replace('min','')), f'Trying to apply a a {args.freq} temporal aggregation while the minimal possible one is {FREQ}'
        df = df.resample(args.freq).mean()

    coverage_local = pd.date_range(start=START, end=END, freq=args.freq)[:-1]
    df = df.reindex(coverage_local)



    df = restrain_df_to_specific_period(df,coverage_period)
    time_step_per_hour = get_time_step_per_hour(args.freq)
    assert time_step_per_hour == 12, 'TIME STEP PER HOUR = {time_step_per_hour} ALORS QU ON VEUT =12 '


    args_DataSet = filter_args(DataSet, args)

    dataset = DataSet(df,
                      time_step_per_hour=time_step_per_hour, 
                      spatial_unit = df.columns,
                      indices_spatial_unit = np.arange(len(df.columns)),
                      dims = [0],       DEFINIR 'DIMS'  // MODIFIER ICIIIIIIII
                      **args_DataSet)


    
    return(dataset)
    
def restrain_df_to_specific_period(df,coverage_period):
    if coverage_period is not None:
        df = df.loc[coverage_period]

    df = df.sort_index()
    return df