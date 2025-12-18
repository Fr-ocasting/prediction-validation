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
NAME = 'bike_in'
FILE_BASE_NAME = 'velov'
DIRECTION = 'attracted' # attracted
FILE_PATTERN = f'{FILE_BASE_NAME}_{DIRECTION}_by_station' # Sera complété par args.freq
DATA_SUBFOLDER = f'agg_data/{FILE_BASE_NAME}' # Sous-dossier dans FOLDER_PATH
CITY = 'Lyon'

# Couverture théorique
START = '2019-01-01' 
END = '2020-01-01'
USELESS_DATES = {'hour':[1,2,3,4,5,6], #[1,2,3,4,5,6],  #[] if no useless (i.e removed) hours
                 'weekday':[]#[5,6],
                 }
# Liste des périodes invalides
list_of_invalid_period = []

C = 1 # Nombre de canaux/features par unité spatiale

# Colonnes attendues dans le CSV
DATE_COL = 'date_retour' 
LOCATION_COL = 'id_retour' # Ou 'id_entree' pour 'attracted'?
VALUE_COL = 'volume'

def load_data(FOLDER_PATH, coverage_period, invalid_dates, args, minmaxnorm,standardize, normalize=True,
              tensor_limits_keeper = None,
              file_pattern = FILE_PATTERN,
              data_subfolder = DATA_SUBFOLDER,
              date_col = DATE_COL,
              location_col = LOCATION_COL,
              value_col = VALUE_COL,
              direction = DIRECTION,
              name = NAME):
    """
    Charge, pivote, filtre et pré-traite les données velov (emitted).
    """
    target_freq = args.freq
    # Construction spécifique du nom de fichier pour velov
    file_name = f"{file_pattern}{target_freq}"
    data_file_path = os.path.join(FOLDER_PATH, data_subfolder, f"{file_name}.csv")

    print(f"Loading from {data_file_path}...")
    try:
        df = pd.read_csv(data_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"ERROR : file {data_file_path} does not exists.\nPlease check path and  frequency '{target_freq}', it has to exists for velov_{direction}")
    except Exception as e:
        raise ImportError(f"ERROR while loading {file_name}.csv: {e}")

    # --- Preprocessing ---
    df[date_col] = pd.to_datetime(df[date_col])
    df_pivoted = df.pivot_table(index=date_col, columns=location_col, values=value_col, aggfunc='sum')

    # Fill Nan value
    df_pivoted = df_pivoted.fillna(0)

    # Convert into Datetime
    df_pivoted.index = pd.to_datetime(df_pivoted.index)
    print('df pivoted: ',df_pivoted.shape)
    df_pivoted = df_pivoted.reindex(pd.date_range(start =START, end = END, freq=target_freq)[:-1]).fillna(0)
    print('df reindexed : ',df_pivoted.shape)
    print('Len coverage period: ',len(coverage_period))
    # Temporal filtering on 'coverage_period'
    df_filtered = df_pivoted[df_pivoted.index.isin(coverage_period)].copy()
    print('df filtered: ',df_filtered.shape)

    if df_filtered.empty:
            raise ImportError(f"ERRROR : not any remaining data after temporal filtering on {file_name}.csv.\nPlease check dataset_coverage: {args.dataset_coverage} and dataset_names {args.dataset_names}")
    print(f"   Loaded data: {df_filtered.shape}")

    # Filtering outliers : 
    df_filtered = remove_outliers_based_on_quantile(df_filtered,args,name)

    
    
    # Spatial Agg: 
    if (name in args.contextual_kwargs.keys()) and ('loading_contextual_data' in args.contextual_kwargs[name].keys()) and args.contextual_kwargs[name]['loading_contextual_data']:
         if ('agg_iris_target_n' in args.contextual_kwargs[name].keys()) and (args.contextual_kwargs[name]['agg_iris_target_n'] is not None):
            target_n = args.contextual_kwargs[name]['agg_iris_target_n']
         if ('threshold_volume_min' in args.contextual_kwargs[name].keys()) and (args.contextual_kwargs[name]['threshold_volume_min'] is not None):
            threshold_volume_min = args.contextual_kwargs[name]['threshold_volume_min']
    elif (name in args.target_kwargs.keys()) :
        if ('agg_iris_target_n' in args.target_kwargs[name].keys()) and (args.target_kwargs[name]['agg_iris_target_n'] is not None):
            target_n = args.target_kwargs[name]['agg_iris_target_n']
        if ('threshold_volume_min' in args.target_kwargs[name].keys()) and (args.target_kwargs[name]['threshold_volume_min'] is not None):
            threshold_volume_min = args.target_kwargs[name]['threshold_volume_min']
    else:
        raise ValueError(f"ERROR: {name} not in args.target_kwargs (keys: {args.target_kwargs.keys()})neither in args.contextual_kwargs (keys: {args.contextual_kwargs.keys()})")
    if 'target_n' not in locals():
        target_n = None
    if 'threshold_volume_min' not in locals():
        threshold_volume_min = None

    if target_n is not None:
        #Load Data: 
        s_zone2stations_path = f"{FOLDER_PATH}/lyon_iris_agg{target_n}/zone2stations.csv"
        s_zone2stations = pd.read_csv(s_zone2stations_path,index_col = 0)

        agg_df = pd.DataFrame(columns = s_zone2stations.index)
        for idx,row in s_zone2stations.iterrows():
            station_id = row.STATION
            columns = list(map(int,station_id.split(' ')))
            effective_columns = [c for c in columns if c in df_filtered.columns]
            agg_df[idx] = df_filtered[effective_columns].sum(axis=1)
    else:
        agg_df = df_filtered.copy()

    if threshold_volume_min is not None:
        mask = agg_df.mean() > threshold_volume_min
        kept_zones = list(mask[mask].index)
        df_filtered = agg_df.T[mask].T
        print(f"   Dimension after spatial agg: {df_filtered.shape}")
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

