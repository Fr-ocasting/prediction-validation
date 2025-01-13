import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
import pandas as pd
from dataset import DataSet

''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - add argument 'n_vertex', 'C' to the NameSpace. These are specific to this data
 - 'time_step_per_hour' which has to be perfectly set
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''

FILE_NAME = 'data_bidon/data_bidon' #.csv
INVALID_DATES = []
C = 1
n_vertex = 10
COVERAGE =  pd.date_range(start='03/16/2019', end='06/1/2019', freq='15min')[:-1][:1000]

def load_data(args,ROOT,FOLDER_PATH,coverage_period = None):
    '''Load the dataset. Supposed to coontains pd.DateTime Index as index, and named columns.
    columns has to represent the spatial units.

    outputs: 
    ---------
    df: contains 
    df.index : coverage period of the dataset 
    invalid_dates : list of invalid dates 
    '''

    df = pd.read_csv(f"{ROOT}/{FOLDER_PATH}/{FILE_NAME}.csv",index_col = 0)
    df.columns.name = 'Station'
    df.index = pd.to_datetime(df.index)

    
    df = restrain_df_to_specific_period(df,coverage_period)
    time_step_per_hour = (60*60)/(df.index[1] - df.index[0]).seconds

    dataset = DataSet(df,
                      time_step_per_hour=time_step_per_hour, 
                      Weeks = args.W, 
                      Days = args.D, 
                      historical_len= args.H,
                      step_ahead=args.step_ahead,
                      spatial_unit = df.columns,
                      data_augmentation= args.data_augmentation
                      )
    
    return(dataset)


def restrain_df_to_specific_period(df,coverage_period):
    if coverage_period is not None:
        df = df.loc[coverage_period]

    df = df.sort_index()
    return df
