import sys
import os
import pandas as pd
import torch
import numpy as np
from datetime import datetime
import h5py

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Personnal import
from dataset import DataSet, PersonnalInput
from build_inputs.load_preprocessed_dataset import load_input_and_preprocess
from utils.utilities import restrain_df_to_specific_period
"""
PEMS08 Dataset
df.shape:  (17856, 170)
START: 
END: 
from https://github.com/RWLinno/ST-LoRA/data
"""
FILE_BASE_NAME = '8'
YEAR = 2016
DATA_SUBFOLDER = f'PEMS0{FILE_BASE_NAME}' 
CITY = f'California_{DATA_SUBFOLDER}'
NAME = f"PEMS0{FILE_BASE_NAME}"
# Naive Freq
NATIVE_FREQ = '5min'
# Temporal Coverage period
START = '2016-07-01 00:00:00'
END = '2016-09-01 00:00:00'
USELESS_DATES = {'hour':[], #[1,2,3,4,5,6],  #[] if no useless (i.e removed) hours
                 'weekday':[]#[5,6],
                 }

# List of invalid period 
list_of_invalid_period = []

C = 1 # Nb channels by spatial units
CHANNELS = ['flow', 'occupancy', 'speed'] # C channels


def load_data(FOLDER_PATH, invalid_dates, coverage_period, args, normalize=True,
              data_subfolder = DATA_SUBFOLDER,
            year = YEAR,
            city =CITY,
            name=NAME,
            channel_name='flow'):
     
    """
    Load data

    Args:
        

    Returns:
        PersonnalInput: Objet contenant les données traitées.
    """
    dirname = f"{FOLDER_PATH}/{data_subfolder}"
    raw_data_path = f"{dirname}/{NAME}.npz"
    print(f"   Load data from: {dirname}")
    data = np.load(raw_data_path)["data"][:,:,CHANNELS.index(channel_name)]
    index_t  = np.load(f"{dirname}/{NAME}_time.npy")
    df = pd.DataFrame(data, index=index_t)

    assert '5min' == args.freq, f"Trying to apply a a {args.freq} temporal aggregation while PeMS is designed for 5min"
    #assert args.D == 0, f"Trying to look {args.D}Day before but there are no Weekends"

    # --- Preprocess ---
    df.index = pd.to_datetime(df.index, unit='ns')
    df = restrain_df_to_specific_period(df,coverage_period)

    if (hasattr(args,'set_spatial_units')) and (args.set_spatial_units is not None) :
        print('   Number of Considered Spatial-Unit: ',len(args.set_spatial_units))
        spatial_unit = args.set_spatial_units
        indices_spatial_unit = [list(df.columns).index(station_i) for station_i in  spatial_unit]
        df = df[spatial_unit]
    else:
        spatial_unit = df.columns
        indices_spatial_unit = np.arange(len(df.columns))

    print(f"   Data loaded with shape: {df.shape}")
    data_T = torch.tensor(df.values).float()
    dims = [0] # if [0] then Normalisation on temporal dim

    processed_input = load_input_and_preprocess(dims = dims,normalize=normalize,invalid_dates=invalid_dates,args=args,data_T=data_T,coverage_period=coverage_period,name=name)

    # --- Finalisation Métadonnées ---
    processed_input.spatial_unit = df.columns.tolist()
    processed_input.C = C
    processed_input.adj_mx_path = f"{dirname}/adj_PEMS08.pkl"
    processed_input.distance_mx_path = f"{dirname}/adj_PEMS08_distance.pkl"  # Raw distance. Has to be normalized with Gaussian Kernel.
    processed_input.raw_data_path =raw_data_path
    processed_input.city = city
    processed_input.indices_spatial_unit = indices_spatial_unit
    # processed_input.periods = None 
    return processed_input


if __name__ == "__main__":
    # blabla
    blabla