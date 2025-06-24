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
from utils.utilities import filter_args,get_time_step_per_hour

''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - add argument 'num_nodes', 'C' to the NameSpace. These are specific to this data
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''

FILE_NAME = 'CRITER_3lanes/CRITER_3lanes'
START = '03/01/2019'
END = '06/01/2019'
USELESS_DATES = {'hour':[], #[1,2,3,4,5,6],  #[] if no useless (i.e removed) hours
                 'weekday':[]#[5,6],
                 }
FREQ = '6min'
list_of_invalid_period = []
#list_of_invalid_period.append([datetime(2019,1,10,15,30),datetime(2019,1,14,15,30)])

#C = 1
#num_nodes = 

def load_csvs(args,FOLDER_PATH,coverage_period,limit_max_nan=200,taux_heure_limit = 100):
    # Load df: 
    df = pd.DataFrame()
    idptm_list = []
    for month_name in ['Mars','Avril','Mai']:
        df_i = pd.read_csv(f"{FOLDER_PATH}/{FILE_NAME}_{month_name}.csv",index_col = 0)
        df_i.HORODATE = pd.to_datetime(df_i.HORODATE)
        forbidden_ids = df_i[df_i.TAUX_HEURE > taux_heure_limit].ID_POINT_MESURE.unique()
        init_idptm = list(df_i.ID_POINT_MESURE.unique())
        remaining_idptm = [idptm for idptm in init_idptm if not idptm in forbidden_ids]
        idptm_list.append(remaining_idptm)
        df_i = df_i.groupby(['ID_POINT_MESURE',pd.Grouper(key = 'HORODATE',freq=args.freq)]).mean()

        
        df = pd.concat([df,df_i])

    df = df.reset_index()

    idptm_list = list(set.intersection(*map(set, idptm_list)))
    df = df[df.ID_POINT_MESURE.isin(idptm_list)]
    df = restrain_df_to_specific_period(df,coverage_period)
    df_loop_occupancy_rate = df.pivot_table(index = 'HORODATE',columns = 'ID_POINT_MESURE',values = 'TAUX_HEURE').sort_index()
    df_flow = df.pivot_table(index = 'HORODATE',columns = 'ID_POINT_MESURE',values = 'DEBIT_HEURE').sort_index()


    df_loop_occupancy_rate_full,df_occupancy_with_nan,nan_too_empty_occupancy,sparse_columns_occupancy = remove_sparse_sensor(df_loop_occupancy_rate,limit_max_nan)
    df_flow_full,df_flow_with_nan,nan_too_empty_flow,sparse_columns_flow = remove_sparse_sensor(df_flow,limit_max_nan)
    return df_loop_occupancy_rate_full,df_flow_full,idptm_list


def load_data(FOLDER_PATH,invalid_dates,coverage_period,args,minmaxnorm,standardize,normalize= True):
    # Load df: 
    df_loop_occupancy_rate,df_flow,idptm_list = load_csvs(args,FOLDER_PATH,coverage_period=coverage_period,limit_max_nan = 200)

    for df_feature_i,name_i in zip([df_loop_occupancy_rate,df_flow],['loop_occupancy_rate','flow']):
        df_feature_i.columns.name = 'sensor'
        
        if (hasattr(args,'set_spatial_units')) and (args.set_spatial_units is not None) :
            spatial_unit = args.set_spatial_units
            indices_spatial_unit = [list(df_feature_i.columns).index(station_i) for station_i in  spatial_unit]
            df_feature_i = df_feature_i[spatial_unit]
        else:
            spatial_unit = df_feature_i.columns
            indices_spatial_unit = np.arange(len(df_feature_i.columns))

        time_step_per_hour = get_time_step_per_hour(args.freq)
        weekly_period =  int((24-len(USELESS_DATES['hour']))*(7-len(USELESS_DATES['weekday']))*time_step_per_hour)
        daily_period =  int((24-len(USELESS_DATES['hour']))*time_step_per_hour)
        periods = [weekly_period,daily_period]  

        args_DataSet = filter_args(DataSet, args)

        globals()[f"dataset_{name_i}"] = DataSet(df_feature_i,
                        time_step_per_hour=time_step_per_hour, 
                        spatial_unit = spatial_unit,
                        indices_spatial_unit = indices_spatial_unit,
                        dims = [0],
                        city = 'Lyon',
                        periods = periods,
                        **args_DataSet)
    #return globals()[f"dataset_loop_occupancy_rate"],globals()[f"dataset_flow"]
    raise NotImplementedError('DEVRAIT ETRE PersonnalInput.preprocess')
    return globals()[f"dataset_loop_occupancy_rate"]
    

def restrain_df_to_specific_period(df,coverage_period):
    if coverage_period is not None:
        df = df[df.HORODATE.isin(coverage_period)]
    return df


def remove_sparse_sensor(df,limit_max_nan = 200):
    df_with_nan = pd.DataFrame()
    for c in df.columns:
        if df[c].isna().sum() > 0:
            df_with_nan[c] = df[c]

    s_nb_nan_per_columns = df_with_nan.isna().sum()
    sparse_columns = s_nb_nan_per_columns[s_nb_nan_per_columns>limit_max_nan].index

    df = df.drop(columns = df_with_nan.columns)
    nan_too_empty = df_with_nan[sparse_columns]
    df_with_nan = df_with_nan.drop(columns = sparse_columns)
    return df,df_with_nan,nan_too_empty,sparse_columns