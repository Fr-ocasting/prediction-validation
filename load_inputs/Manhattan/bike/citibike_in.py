import sys
import os
import pandas as pd
import torch
import numpy as np
from datetime import datetime
import pickle 

# --- Gestion de l'arborescence ---
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# --- Importations personnalisées ---
from pipeline.dataset import DataSet, PersonnalInput
from pipeline.utils.utilities import filter_args,remove_outliers_based_on_quantile # Assurez-vous que ce chemin est correct
from pipeline.build_inputs.load_preprocessed_dataset import load_input_and_preprocess
# --- Constantes spécifiques à cette donnée ---
YEAR_MIN = 2022  # 2019 / 2020 / 2021 / 2022 / 2023
YEAR_MAX = 2022
MONTH_MIN = 4
MONTH_MAX = 10

NAME = 'citibike_in'
FILE_BASE_NAME = 'citibike'
DIRECTION = 'attracted' # attracted
CITY = 'Manhattan'


# Couverture théorique
START = f'{YEAR_MIN}-{MONTH_MIN:02d}-01' 
END = f'{YEAR_MAX}-{MONTH_MAX:02d}-01'
USELESS_DATES = {'hour':[], #[1,2,3,4,5,6],  #[] if no useless (i.e removed) hours
                 'weekday':[]#[5,6],
                 }
# Liste des périodes invalides
list_of_invalid_period = []

C = 1 # Nombre de canaux/features par unité spatiale


def load_bike_data_from_csv(save_path, year, month, target_freq, direction):
    if (target_freq == 'H') or (target_freq == '1H') or (target_freq == 'h'):
        target_freq_str = '1h'
    else:
        target_freq_str = target_freq

    if direction == 'emitted':
        key_time = 'started_at'
    elif direction == 'attracted':
        key_time = 'ended_at'
    path = f"{save_path}/city_bike_{year}/{month:02d}_{target_freq_str}_{direction}.csv"
    df_bike = pd.read_csv(path,index_col=0,dtype={0: str, 1:str, 2:int}) # specify columns dtypes
    df_bike = df_bike.rename(columns={'starttime': 'started_at', 'stoptime': 'ended_at'})
    df_bike.index.name ='station_id'
    df_bike[key_time] = pd.to_datetime(df_bike[key_time])
    nb_bike_stations = df_bike.index.nunique()

    # Pivoted bike:
    df_pivoted_bike_i = df_bike.reset_index().pivot_table(index=key_time, columns='station_id', values='Flow').fillna(0).sort_index()
    df_pivoted_bike_i = df_pivoted_bike_i[df_pivoted_bike_i.index > datetime(year,month,1)]

    return df_pivoted_bike_i, nb_bike_stations

def load_data(FOLDER_PATH, coverage_period, invalid_dates, args, minmaxnorm,standardize, normalize=True,
              tensor_limits_keeper = None,
              direction = DIRECTION,
              name = NAME):
    """
    Charge, pivote, filtre et pré-traite les données velov (emitted).
    """
    target_freq = args.freq

    # =================== Load Datasets ============================
    All_df_pivoted_bike = []
    All_nb_bike_stations = []
    save_path = os.path.join(FOLDER_PATH,CITY)
    for year in range(YEAR_MIN,YEAR_MAX+1):
        print("Loading from" ,f"{save_path}/city_bike_{year}/")
        for month in range(1,13):
            # 01_15min_emitted.csv
            df_pivoted_bike_i, nb_bike_stations = load_bike_data_from_csv(save_path, year, month, target_freq, direction)

            All_df_pivoted_bike.append(df_pivoted_bike_i)
            All_nb_bike_stations.append(nb_bike_stations)
    df_pivoted = pd.concat(All_df_pivoted_bike).sort_index().copy()
    df_pivoted = df_pivoted.fillna(0)
    # ============================================================
   

    print('df pivoted: ',df_pivoted.shape)
    df_pivoted = df_pivoted.reindex(pd.date_range(start =START, end = END, freq=target_freq.replace('H','h'))[:-1]).fillna(0)
    print('df reindexed : ',df_pivoted.shape)
    print('Len coverage period: ',len(coverage_period))
    # Temporal filtering on 'coverage_period'
    df_filtered = df_pivoted[df_pivoted.index.isin(coverage_period)].copy()
    print('df filtered: ',df_filtered.shape)


    # Filtering outliers : 
    df_filtered = remove_outliers_based_on_quantile(df_filtered,args,name)

    
    
    # Spatial Agg: 
    if (name in args.contextual_kwargs.keys()) and ('loading_contextual_data' in args.contextual_kwargs[name].keys()) and args.contextual_kwargs[name]['loading_contextual_data']:
        #  if ('agg_iris_target_n' in args.contextual_kwargs[name].keys()) and (args.contextual_kwargs[name]['agg_iris_target_n'] is not None):
        #     target_n = args.contextual_kwargs[name]['agg_iris_target_n']
         if ('threshold_volume_min' in args.contextual_kwargs[name].keys()) and (args.contextual_kwargs[name]['threshold_volume_min'] is not None):
            threshold_volume_min = args.contextual_kwargs[name]['threshold_volume_min']
    elif (name in args.target_kwargs.keys()) :
        # if ('agg_iris_target_n' in args.target_kwargs[name].keys()) and (args.target_kwargs[name]['agg_iris_target_n'] is not None):
        #     target_n = args.target_kwargs[name]['agg_iris_target_n']
        if ('threshold_volume_min' in args.target_kwargs[name].keys()) and (args.target_kwargs[name]['threshold_volume_min'] is not None):
            threshold_volume_min = args.target_kwargs[name]['threshold_volume_min']
    else:
        raise ValueError(f"ERROR: {name} not in args.target_kwargs (keys: {args.target_kwargs.keys()})neither in args.contextual_kwargs (keys: {args.contextual_kwargs.keys()})")
    # if 'target_n' not in locals():
    #     target_n = None
    if 'threshold_volume_min' not in locals():
        threshold_volume_min = None

    # if target_n is not None:
    #     #Load Data: 
    #     s_zone2stations_path = f"{FOLDER_PATH}/{manhattan}_area_agg{target_n}/zone2stations.csv"
    #     s_zone2stations = pd.read_csv(s_zone2stations_path,index_col = 0)

    #     agg_df = pd.DataFrame(columns = s_zone2stations.index)
    #     for idx,row in s_zone2stations.iterrows():
    #         station_id = row.STATION
    #         columns = list(map(int,station_id.split(' ')))
    #         effective_columns = [c for c in columns if c in df_filtered.columns]
    #         agg_df[idx] = df_filtered[effective_columns].sum(axis=1)
    # else:
    #     agg_df = df_filtered.copy()

    if threshold_volume_min is not None:
        mask = df_filtered.mean() > threshold_volume_min
        kept_zones = list(mask[mask].index)
        df_filtered = df_filtered.T[mask].T
        print(f"   Dimension after threshold filtering: {df_filtered.shape}")
    else:
        kept_zones = df_filtered.columns.tolist()


    # Convert into tensor:
    data_T = torch.tensor(df_filtered.values).float()  #[T,C]


    dims = [0] # if [0] then Normalisation on temporal dim
    processed_input = load_input_and_preprocess(dims = dims,normalize=normalize,invalid_dates=invalid_dates,
                                                args=args,data_T=data_T,coverage_period=coverage_period,
                                                freq = target_freq, step_ahead = args.step_ahead, horizon_step=args.horizon_step,
                                                name=name,minmaxnorm=minmaxnorm,standardize=standardize,
                                                tensor_limits_keeper = tensor_limits_keeper)
    processed_input.spatial_unit = df_filtered.columns.tolist()
    processed_input.C = C
    processed_input.periods = None 
    processed_input.kept_zones = kept_zones
    processed_input.city = CITY
    return processed_input

