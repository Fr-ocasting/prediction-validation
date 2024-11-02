import sys 
import os 
import pandas as pd
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from dataset import DataSet
from datetime import datetime 
from build_inputs.preprocess_subway_15 import get_trigram_correspondance

''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - add argument 'n_vertex', 'C' to the NameSpace. These are specific to this data
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''

FILE_NAME = 'subway_IN_interpol_neg_15_min_2019_2020' #.csv

list_of_invalid_period = []
list_of_invalid_period.append([datetime(2019,1,10,15,30),datetime(2019,1,14,15,30)])
list_of_invalid_period.append([datetime(2019,1,30,8,15),datetime(2019,1,30,10,30)])
list_of_invalid_period.append([datetime(2019,2,18,11),datetime(2019,2,18,13)])
list_of_invalid_period.append([datetime(2019,4,23,14),datetime(2019,4,28,14)])
list_of_invalid_period.append([datetime(2019,6,26,11),datetime(2019,6,28,4)])
list_of_invalid_period.append([datetime(2019,10,27),datetime(2019,10,28,16)])
list_of_invalid_period.append([datetime(2019,12,21,15,45),datetime(2019,12,21,16,45)])

INVALID_DATES = []
for start,end in list_of_invalid_period:
    INVALID_DATES = INVALID_DATES + list(pd.date_range(start,end,freq = f'15min'))
C = 1
n_vertex = 40
coverage = pd.date_range(start='01/01/2019', end='01/01/2020', freq='15min')[:-1]


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
    time_step_per_hour = (60*60)/(df.iloc[1].name - df.iloc[0].name).seconds

    dataset = DataSet(df,
                      time_step_per_hour=time_step_per_hour, 
                      Weeks = args.W, 
                      Days = args.D, 
                      historical_len= args.H,
                      step_ahead=args.step_ahead,
                      )
    
    df_correspondance = get_trigram_correspondance()
    df_correspondance.set_index('Station').reindex(dataset.spatial_unit)
    dataset.spatial_unit = df_correspondance.COD_TRG
    
    return(dataset)
    
def restrain_df_to_specific_period(df,coverage_period):
    if coverage_period is not None:
        df = df.loc[coverage_period]
    return df