import sys 
import os 
import pandas as pd
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from dataset import DataSet
import h5py

''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - add argument 'n_vertex', 'C' to the NameSpace. These are specific to this data
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''

FILE_NAME = 'METR_LA/METR_LA'#'subway_IN_interpol_neg_15_min_2019_2020' #.csv
C = 1
n_vertex = 207
freq = '5min'
COVERAGE = pd.date_range(start='03/01/2012', end='06/28/2012', freq=freq)[:-1]
list_of_invalid_period = []
#list_of_invalid_period.append([datetime(2019,1,10,15,30),datetime(2019,1,14,15,30)])
#list_of_invalid_period.append([datetime(2019,1,30,8,15),datetime(2019,1,30,10,30)])

INVALID_DATES = []
for start,end in list_of_invalid_period:
    INVALID_DATES = INVALID_DATES + list(pd.date_range(start,end,freq = freq))




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
    axis0 = pd.Series(data['df']['axis0'][:].astype(str))
    axis1 = pd.Series(data['df']['axis1'][:].astype(str))
    df = pd.DataFrame(data['df']['block0_values'][:], columns=axis0, index = pd.to_datetime(axis1.astype(int)/1_000_000_000,unit='s'))
    df.columns.name = 'Sensor'
    df.index = pd.to_datetime(df.index)

    df = restrain_df_to_specific_period(df,coverage_period)
    time_step_per_hour = (60*60)/(df.iloc[1].name - df.iloc[0].name).seconds
    assert time_step_per_hour == 12, 'TIME STEP PER HOUR = {time_step_per_hour} ALORS QU ON VEUT =12 '

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