import sys 
import os 
import pandas as pd
import numpy as np
import torch
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from pipeline.DataSet.dataset import DataSet
from datetime import datetime 
from pipeline.utils.utilities import filter_args,get_time_step_per_hour,restrain_df_to_specific_period,remove_outliers_based_on_quantile
from pipeline.build_inputs.load_preprocessed_dataset import load_input_and_preprocess

''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - add argument 'num_nodes', 'C' to the NameSpace. These are specific to this data
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''
NAME= 'MTA_in'
CITY = 'Manhattan'
YEAR_MIN = 2022  # 2019 / 2020 / 2021 / 2022 / 2023
YEAR_MAX = 2022
MONTH_MIN = 4
MONTH_MAX = 10

START = f'{YEAR_MIN}-{MONTH_MIN:02d}-01' 
END = f'{YEAR_MAX+1}-{MONTH_MAX:02d}-01'
FREQ = '1H'

USELESS_DATES = {'hour':[],  #[] if no useless (i.e removed) hours
                 'weekday':[]#[5,6],
                 }


list_of_invalid_period = []
# list_of_invalid_period.append([datetime(2019,1,10,15,30),datetime(2019,1,14,15,30)])
# list_of_invalid_period.append([datetime(2019,1,30,8,15),datetime(2019,1,30,10,30)])


C = 1

def load_data(FOLDER_PATH,
              invalid_dates,
              coverage_period,
              args,
              minmaxnorm,
              standardize,
              normalize= True,
              name=NAME,
              tensor_limits_keeper = None
              ):
    dataset = load_DataSet(args,FOLDER_PATH,coverage_period = coverage_period,name=name)

    if  hasattr(args,'contextual_kwargs') and (name in args.contextual_kwargs.keys()) and ('use_future_values' in args.contextual_kwargs[name].keys()) and args.contextual_kwargs[name]['use_future_values'] and ('loading_contextual_data' in args.contextual_kwargs[name].keys()) and args.contextual_kwargs[name]['loading_contextual_data']:
        data_T = torch.roll(torch.Tensor(dataset.raw_values), shifts=-1, dims=0)
        print(f">>>>> ICI ON UTILISE LE {name.upper()} IN FUTURE !!!!")
        print('data_T.size: ',data_T.size())
    else:
        data_T = dataset.raw_values

    preprocesed_ds = load_input_and_preprocess(dims = dataset.dims,normalize=normalize,invalid_dates=invalid_dates,args=args,data_T=data_T,
                                            coverage_period=coverage_period,name=name,
                                            minmaxnorm=minmaxnorm,standardize=standardize,
                                            tensor_limits_keeper=tensor_limits_keeper)
    
    preprocesed_ds.spatial_unit = dataset.spatial_unit
    preprocesed_ds.dims = dataset.dims
    preprocesed_ds.periods = dataset.periods
    preprocesed_ds.time_step_per_hour = dataset.time_step_per_hour
    preprocesed_ds.indices_spatial_unit = dataset.indices_spatial_unit
    preprocesed_ds.city = dataset.city


    return preprocesed_ds


def load_DataSet(args,FOLDER_PATH,coverage_period = None,name = NAME):
    '''Load the dataset. Supposed to coontains pd.DateTime Index as index, and named columns.
    columns has to represent the spatial units.

    outputs: 
    ---------
    df: contains 
    df.index : coverage period of the dataset 
    invalid_dates : list of invalid dates 
    '''

    df = load_subway_in_df(args,
                           FOLDER_PATH,
                           coverage_period = coverage_period,
                           name=name)

    if (hasattr(args,'set_spatial_units')) and (args.set_spatial_units is not None) :
        print('   Number of Considered Spatial-Unit: ',len(args.set_spatial_units))
        spatial_unit = args.set_spatial_units
        indices_spatial_unit = [list(df.columns).index(station_i) for station_i in  spatial_unit]
        df = df[spatial_unit]
    else:
        spatial_unit = df.columns
        indices_spatial_unit = np.arange(len(df.columns))

    time_step_per_hour = get_time_step_per_hour(args.freq)
    weekly_period =  int((24-len(USELESS_DATES['hour']))*(7-len(USELESS_DATES['weekday']))*time_step_per_hour)
    daily_period =  int((24-len(USELESS_DATES['hour']))*time_step_per_hour)
    periods = [weekly_period,daily_period]  

    args_DataSet = filter_args(DataSet, args)
    if 'time_step_per_hour' in args_DataSet.keys():
        del args_DataSet['time_step_per_hour']
    


    dataset = DataSet(df,
                      time_step_per_hour=time_step_per_hour, 
                      spatial_unit = spatial_unit,
                      indices_spatial_unit = indices_spatial_unit,
                      dims = [0],
                      city = CITY,
                      periods = periods,
                      **args_DataSet)

    return(dataset)
    

def load_subway_in_df(args,FOLDER_PATH,coverage_period,name):
    
    
    MTA_hourly = []
    for year in range(YEAR_MIN, YEAR_MAX+1):
        print(f"   Load data from: /{FOLDER_PATH}/{CITY}/MTA_Turnstile_Manhattan_hourly_{year}.csv")
        try:
            MTA_hourly_year = pd.read_csv(f'{FOLDER_PATH}/{CITY}/MTA_Turnstile_Manhattan_hourly_{year}.csv', index_col=0)
        except:
            raise FileNotFoundError(f"   ERROR : File {FOLDER_PATH}/{CITY}/MTA_Turnstile_Manhattan_hourly_{year}.csv has not been found.")
        MTA_hourly_year.index = pd.to_datetime(MTA_hourly_year.index)
        MTA_hourly_year = MTA_hourly_year.fillna(0)
        MTA_hourly.append(MTA_hourly_year)
    df = pd.concat(MTA_hourly)
    
    df.columns.name = 'Station'

    # Remove ouliers
    df = remove_outliers(df,args,name)


    if args.freq != FREQ :
        if args.freq == 'H' or  args.freq == 'h':
            freq_i = 60
        elif args.freq[-1] == 'H': 
            freq_i = int(args.freq.replace('H',''))*60
        else:
            freq_i = int(args.freq.replace('min',''))
        assert int(freq_i)>= 60, f'Trying to apply a a {args.freq} temporal aggregation while the minimal possible one is {FREQ}'
        df = df.resample(args.freq).sum()

    # Temporal Restriction: 
    df = restrain_df_to_specific_period(df,coverage_period)
    return df
  
def remove_outliers(df,args,name):
    '''
    Replace the outliers by linear interpolation. Outliers are identified as MaxiMum flow recorded during the 'light festival' in Lyon. 
    It's an atypical event which reach the highest possible flow. Having higher flow on passenger is almost impossible.
    '''

    if False: 
        limits = {
        }
        default_limit = ...

        # Appliquer les limites
        for column in df.columns:
            limit = limits.get(column, default_limit)
            df[column] = df[column].where(df[column] <= limit, None)

        # Interpolation linéaire
        df_interpolated = df.interpolate(method='linear')

        # Remplacer les valeurs originales par les interpolées
        df.update(df_interpolated)

    # # remove outliers by quantile filtering. But differenciate according if it's for contextual dataset or target dataset:
    df = remove_outliers_based_on_quantile(df,args,name)

    return df

