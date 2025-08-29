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
from utils.utilities import remove_outliers_based_on_quantile
NAME = 'weather'
FILE_NAME = 'weather/donnees-meteo'  # 'subway_out/subway_out'  #  'subway_in/subway_in' 
START = '01/01/2019'
END = '01/01/2021'

FREQ = '60min'
USELESS_DATES = {}
list_of_invalid_period = []
C = 1

def load_data(FOLDER_PATH,invalid_dates,coverage_period,args,minmaxnorm,standardize,normalize= True,filename=None,tensor_limits_keeper = None): # args,FOLDER_PATH,coverage_period = None
    '''
    args:
    ------


    outputs:
    --------
    PersonalInput object. Containing a 2-th order tensor [T,R]
    '''
    interpolated_weather = load_preprocessed_weather_df(args,coverage_period,folder_path = FOLDER_PATH)
    data_T = torch.tensor(interpolated_weather.values).float()  # Tensor of shape [T,N]
    
    dims = [0]# [0]  -> We are normalizing each time-serie independantly 
    preprocessed_ds = load_input_and_preprocess(dims = dims,normalize=normalize,invalid_dates=invalid_dates,
                                           args=args,data_T=data_T,coverage_period = coverage_period,
                                           freq = args.freq,step_ahead = args.step_ahead, horizon_step = args.horizon_step,
                                           name=NAME, minmaxnorm=minmaxnorm,
                                           standardize=standardize,
                                           tensor_limits_keeper=tensor_limits_keeper) 
    return preprocessed_ds


def load_preprocessed_weather_df(args,coverage_period,folder_path):
    df_weather = load_raw_weather_df(folder_path = folder_path,
                                    pathway = f"{FILE_NAME}.csv",
                                    id_stations = [69029001,69299001],
                                    columns = ['POSTE','DATE','T','RR1','DRR1','FF','GLO'],
                                    dico = {'POSTE':'id_station','DATE':'date','T':'temperature', 
                            'RR1':'precip','DRR1':'duree_prec','FF':'wind_ms','GLO':'solar'}
                            )
                
    df_weather = df_weather.applymap(lambda x: float(x.replace(',','.')) if isinstance(x,str) else x)
    pivoted_df_weather = df_weather.pivot_table(index='date',columns='id_station',values='precip')

    # Restraint df: 
    coverage_local = pd.date_range(start=START, end=END, freq=args.freq)[:-1]
    if args.freq in ['15min','30min']:
        pivoted_df_weather = pivoted_df_weather.reindex(coverage_local)
    else:
        raise NotImplementedError(f"Frequency {args.freq} not implemented")
    pivoted_df_weather = restrain_df_to_specific_period(pivoted_df_weather,coverage_period)
    pivoted_df_weather = remove_outliers_based_on_quantile(pivoted_df_weather,args,NAME)


    # Interpolation : 
    interpolated_weather = pivoted_df_weather.copy()
    interpolated_weather = interpolated_weather.interpolate(method='polynomial', order=2)
    interpolated_weather[interpolated_weather<1e-3] = 0
    return interpolated_weather


def date_transform(date):
    date = ''.join('-'.join('-'.join(date.split('/')).split(':')).split(' '))
    date = datetime.strptime(date, '%d-%m-%Y-%H')
    date = date.strftime('%Y-%m-%d-%H')
    return(date)


def load_raw_weather_df(folder_path = 'Data/Meteo',
                     pathway = 'donnees-meteo.csv',
                     id_stations = [69029001,69299001],
                     columns = ['POSTE','DATE','T','RR1','DRR1','FF','GLO'],
                     dico = {'POSTE':'id_station','DATE':'date','T':'temperature', 
            'RR1':'precip','DRR1':'duree_prec','FF':'wind_ms','GLO':'solar'}
            ):
    # Get 'donnees-meteo' :
    df_weather = pd.read_csv(f"{folder_path}/{pathway}",on_bad_lines='skip', sep = ";",low_memory=False,)  # Opening data
    df_weather = df_weather[columns]  # Columns selections 
    df_weather = df_weather.rename(columns = dico)  # Rename 
    df_weather = df_weather[df_weather['id_station'].isin(id_stations)]  # Filter station
    
    df_weather['date'] = df_weather['date'].apply(lambda date : date_transform(date) )  # Date transform 
    df_weather['date'] = pd.to_datetime(df_weather['date'])

    return(df_weather)

def fill_empty_values(df_weather):
    grouped_df = df_weather.groupby('date')   # Group station
    for date,group in grouped_df:    
        for index, row in group.iterrows():
            for column in ['temperature','precip','duree_prec','wind_ms','solar']:
                if pd.isnull(row[column]):
                    df_weather.at[index, column] = group[group.index != index][column].values[0]  #Fill empty values 
    return(df_weather)

def sort_and_interpolate(df_weather):
    df_weather = df_weather.set_index(['id_station','date'])    # Sorr by id and by date
    df_weather = df_weather.sort_values(['id_station','date'])

    df_weather = df_weather.interpolate(method = 'linear',limit_direction='forward',axis = 0)   # Interpolate Empty values 
    
    # Convert french format (32,4) to US format (32.4)
    for column in ['temperature','precip','wind_ms']:
        df_weather[column] =  df_weather[column].apply(lambda t : float('.'.join(t.split(','))))
    return(df_weather)


if __name__ == "__main__":

    import os
    import sys 
    current_path = os.path.dirname(os.path.abspath(__file__))
    working_dir = os.path.abspath(os.path.join(current_path, '..'))
    if working_dir not in sys.path:
        sys.path.insert(0, working_dir)

    from constants.paths import FOLDER_PATH

    folder_path = f"{FOLDER_PATH}/weather"
    csv_path = 'donnees-meteo.csv'
    id_stations = [69029001,69299001]  # Lyon Bron and Lyon St-exupery
    columns = ['POSTE','DATE','T','RR1','DRR1','FF','GLO']
    dico = {'POSTE':'id_station','DATE':'date','T':'temperature', 
            'RR1':'precip','DRR1':'duree_prec','FF':'wind_ms','GLO':'solar'}

    df_weather = load_raw_weather_df(folder_path,csv_path,id_stations=id_stations,columns=columns,dico=dico)
    print(df_weather.head())