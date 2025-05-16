import sys 
import os 
import pandas as pd
import torch 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from utils.utilities import restrain_df_to_specific_period
from build_inputs.load_preprocessed_dataset import load_input_and_preprocess

''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - add argument 'num_nodes', 'C' to the NameSpace. These are specific to this data
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''

DATA_SUBFOLDER = 'CRITER_3_4_5_lanes'
NAME = 'flow'
START = '03/01/2019'
END = '06/01/2019'
USELESS_DATES = {'hour':[], #[1,2,3,4,5,6],  #[] if no useless (i.e removed) hours
                 'weekday':[]#[5,6],
                 }
FREQ = '6min'
list_of_invalid_period = []
#list_of_invalid_period.append([datetime(2019,1,10,15,30),datetime(2019,1,14,15,30)])

C = 1
CITY = 'Lyon'
CHANNEL = 'DEBIT_HEURE'

def load_data(FOLDER_PATH,invalid_dates,coverage_period,args,normalize= True,
              data_subfolder = DATA_SUBFOLDER,
              name=NAME,
              channel = CHANNEL,
              city = CITY,
              ): # args,FOLDER_PATH,coverage_period = None
    assert '6min' == args.freq, f"Trying to apply a a {args.freq} temporal aggregation while CRITER is designed for 6min"

    pivoted_df_full,idptm_list = load_csvs(args,FOLDER_PATH,coverage_period,data_subfolder = data_subfolder,channel = channel,limit_max_nan=200,taux_heure_limit = 100)
    data_T = torch.tensor(pivoted_df_full.values).float()
    dims = [0]# [0]  -> We are normalizing each time-serie independantly 


    processed_input = load_input_and_preprocess(dims = dims,normalize=normalize,invalid_dates=invalid_dates,args=args,data_T=data_T,coverage_period = coverage_period,name=f"{DATA_SUBFOLDER}_{name}") 

    # --- Finalisation Métadonnées ---
    processed_input.spatial_unit = pivoted_df_full.columns.tolist()
    processed_input.C = C
    processed_input.adj_mx_path = f"{FOLDER_PATH}/{data_subfolder}/{data_subfolder}_adj.npy"
    processed_input.distance_mx_path = f"{FOLDER_PATH}/{data_subfolder}/{data_subfolder}_adj.npy"
    processed_input.raw_data_path =f"{FOLDER_PATH}/{data_subfolder}/{data_subfolder}.csv"
    processed_input.city = city
    return processed_input



def load_csvs(args,FOLDER_PATH,coverage_period,data_subfolder,channel,limit_max_nan=200,taux_heure_limit = 100):
    """
    Load the csv files for the specified month and filter the data based on the given criteria.
    Args:
        args: Namespace object containing the arguments.
        FOLDER_PATH: Path to the folder containing the CSV files.
        coverage_period: List of datetime objects representing the coverage period.
        limit_max_nan: Maximum number of NaN values allowed in a column.
        taux_heure_limit: Threshold for filtering based on TAUX_HEURE. Within [0 - 100].
        channel: The channel to be used for pivoting the DataFrame.
    Returns:
        pivoted_df_full: Pivoted DataFrame with the specified channel.
        idptm_list: List of unique ID_POINT_MESURE values after filtering.
    """

    # Load df: 
    df = pd.read_csv(f"{FOLDER_PATH}/{data_subfolder}/{data_subfolder}.csv",index_col = 0)
    df.HORODATE = pd.to_datetime(df.HORODATE)

    print('Number of Init sensors: ',len(df.ID_POINT_MESURE.unique()))

    forbidden_ids = df[(df.TAUX_HEURE > taux_heure_limit) | (df.DEBIT_HEURE < 0)].ID_POINT_MESURE.unique()
    init_idptm = list(df.ID_POINT_MESURE.unique())
    idptm_list = [idptm for idptm in init_idptm if not idptm in forbidden_ids]

    df = df.groupby(['ID_POINT_MESURE',pd.Grouper(key = 'HORODATE',freq=args.freq)]).mean()
    df = df.reset_index()

    df = df[df.ID_POINT_MESURE.isin(idptm_list)]
    df = restrain_df_to_specific_period(df,coverage_period)
    pivoted_df = df.pivot_table(index = 'HORODATE',columns = 'ID_POINT_MESURE',values =channel).sort_index()

    print('Number of sensors after filter on TAUX_HEURE and "DEBIT_HEURE : ',len(pivoted_df.columns))


    pivoted_df_full,df_with_nan,nan_too_empty,sparse_columns = remove_sparse_sensor(pivoted_df,limit_max_nan)
    print('Number of sensors after filter sparse sensor : ',len(pivoted_df_full.columns))
    print(f" Data loaded with shape: {pivoted_df_full.shape}")
    return pivoted_df_full,idptm_list


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