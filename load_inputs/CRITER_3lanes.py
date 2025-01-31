import sys 
import os 
import pandas as pd
import numpy as np
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from dataset import DataSet
from datetime import datetime 
from utils.utilities import filter_args
from constants.paths import USELESS_DATES
''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - add argument 'n_vertex', 'C' to the NameSpace. These are specific to this data
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''

FILE_NAME = 'CRITER_3lanes/CRITER_3lanes'

list_of_invalid_period = []
#ist_of_invalid_period.append([datetime(2019,1,10,15,30),datetime(2019,1,14,15,30)])


INVALID_DATES = []
for start,end in list_of_invalid_period:
    INVALID_DATES = INVALID_DATES + list(pd.date_range(start,end,freq = f'30min'))
C = 1
n_vertex = 40
COVERAGE = pd.date_range(start='03/01/2019', end='06/01/2020', freq='30min')[:-1]

coverage_period = None
freq = '30min'
time_step_per_hour = 2


def load_data(args,ROOT,FOLDER_PATH,coverage_period = None):
    df = pd.DataFrame()
    for month_name in ['Mars','Avril','Mai']:
        df_i = pd.read_csv(f"{ROOT}/{FOLDER_PATH}/{FILE_NAME}_{month_name}.csv",index_col = 0)
        df_i.HORODATE = pd.to_datetime(df_i.HORODATE)
        df_i = df_i.groupby(['ID_POINT_MESURE',pd.Grouper(key = 'HORODATE',freq=freq)]).mean()
        df = pd.concat([df,df_i])
        df = df.reset_index()
        if coverage_period is not None:
            df = df[(df.HORODATE <= coverage_period.max())&(df.HORODATE >= coverage_period.min()) ]
        df_loop_occupancy_rate = df.pivot_table(index = 'HORODATE',column = 'ID_POINT_MESURE',value = 'TAUX_HEURE')
        df_flow = df.pivot_table(index = 'HORODATE',column = 'ID_POINT_MESURE',value = 'DEBIT_HEURE')

        for df_i,name_i in zip([df_loop_occupancy_rate,df_flow],['loop_occupancy_rate','flow']):
            df_i.columns.name = 'sensor'
            
            if (hasattr(args,'set_spatial_units')) and (args.set_spatial_units is not None) :
                print('Considered Spatial-Unit: ',args.set_spatial_units)
                spatial_unit = args.set_spatial_units
                indices_spatial_unit = [list(df_i.columns).index(station_i) for station_i in  spatial_unit]
                df_i = df_i[spatial_unit]
            else:
                spatial_unit = df_i.columns
                indices_spatial_unit = np.arange(len(df_i.columns))

            weekly_period =  int((24-len(USELESS_DATES['hour']))*(7-len(USELESS_DATES['weekday']))*time_step_per_hour)
            daily_period =  int((24-len(USELESS_DATES['hour']))*time_step_per_hour)
            periods = [weekly_period,daily_period]  

            args_DataSet = filter_args(DataSet, args)

            globals()[f"ataset_{name_i}"] = DataSet(df_i,
                            time_step_per_hour=time_step_per_hour, 
                            spatial_unit = spatial_unit,
                            indices_spatial_unit = indices_spatial_unit,
                            dims = [0],
                            city = 'Lyon',
                            periods = periods,
                            **args_DataSet)
        return globals()[f"dataset_loop_occupancy_rate"],globals()[f"dataset_flow"]
    