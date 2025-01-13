import sys 
import os 
import pandas as pd
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
from datetime import datetime 
from dataset import DataSet

''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - add argument 'n_vertex', 'C' to the NameSpace. These are specific to this data
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''

#FILE_NAME = 'netmob....' #.csv
list_of_invalid_period = []
list_of_invalid_period.append([datetime(2019,5,16,0,0),datetime(2019,5,16,18,15)])  # 16 mai 00:00 - 18:15
list_of_invalid_period.append([datetime(2019,5,11,23,15),datetime(2019,5,12,0,0)])  # 11 mai 23:15 - 11 mai 23:59: down META (fb, whatsapp)
list_of_invalid_period.append([datetime(2019,5,23,0,0),datetime(2019,5,25,6,0)])  # Anoamlies for every single apps  23-25 May


INVALID_DATES = []
for start,end in list_of_invalid_period:
    INVALID_DATES = INVALID_DATES + list(pd.date_range(start,end,freq = f'15min'))

C = 1
n_vertex = 40
COVERAGE = pd.date_range(start='03/16/2019', end='06/1/2019', freq='15min')[:-1]


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

    print("\n>>>>> iloc de 0: ",df.iloc[1].name)
    print(">>>>> iloc de 1: ",df.iloc[0].name)
    print(">>>>> différence des deux: ",df.iloc[1].name - df.iloc[0].name)
    print(">>>>> Time step per hour: ",time_step_per_hour)
    raise ValueError('VERIFIER ICI QU ON A BIEN TIME STEP PER HOUR = 4 ')

    dataset = DataSet(df,
                      time_step_per_hour=time_step_per_hour, 
                      Weeks = args.W, 
                      Days = args.D, 
                      historical_len= args.H,
                      step_ahead=args.step_ahead,
                      data_augmentation= args.data_augmentation
                      )
    
    df_correspondance = get_trigram_correspondance()
    df_correspondance.set_index('Station').reindex(dataset.spatial_unit)
    dataset.spatial_unit = df_correspondance.COD_TRG
    
    return(dataset)
    
def restrain_df_to_specific_period(df,coverage_period):
    if coverage_period is not None:
        df = df.loc[coverage_period]

    df = df.sort_index()
    return df