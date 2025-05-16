import sys
import os
import pandas as pd
import torch
import h5py

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Personnal import
from build_inputs.load_preprocessed_dataset import load_input_and_preprocess
from utils.utilities import restrain_df_to_specific_period

"""
PEMS03 Dataset
df.shape:  (26208, 358)
from https://github.com/RWLinno/ST-LoRA/data
"""
FILE_BASE_NAME = '3'
YEAR = 2018
DATA_SUBFOLDER = f'PEMS0{FILE_BASE_NAME}' 
CITY = f'California_{DATA_SUBFOLDER}'
NAME = "PEMS03"

# Naive Freq
NATIVE_FREQ = '5min'
# Temporal Coverage period
START = '2018-09-01 00:00:00' 
END = '2018-12-01 00:00:00'
USELESS_DATES = {'hour':[], #[1,2,3,4,5,6],  #[] if no useless (i.e removed) hours
                 'weekday':[]#[5,6],
                 }
# List of invalid period 
list_of_invalid_period = []

C = 1 # Nb channels by spatial units


def load_data(FOLDER_PATH, invalid_dates, coverage_period, args, normalize=True,
              data_subfolder = DATA_SUBFOLDER,
            year = YEAR,
            city =CITY,
            name=NAME):
     
    """
    Load data

    Args:
        

    Returns:
        PersonnalInput: Objet contenant les données traitées.
    """

    dirname = f"{FOLDER_PATH}/{data_subfolder}"
    print(f"   Load data from: {dirname}")

    data = h5py.File(f"{dirname}/{data_subfolder.lower()}_his_{year}.h5", 'r')
    df = pd.DataFrame(data['t']['block0_values'],index = data['t']['axis1'],columns = data['t']['axis0'] )


    assert '5min' == args.freq, f"Trying to apply a a {args.freq} temporal aggregation while PeMS is designed for 5min"
    #assert args.D == 0, f"Trying to look {args.D}Day before but there are no Weekends"

    # --- Preprocess ---
    df.index = pd.to_datetime(df.index, unit='ns')
    df = restrain_df_to_specific_period(df,coverage_period)
    print(f"   Data loaded with shape: {df.shape}")
    data_T = torch.tensor(df.values).float()
    dims = [0] # if [0] then Normalisation on temporal dim

    processed_input = load_input_and_preprocess(dims = dims,normalize=normalize,invalid_dates=invalid_dates,args=args,data_T=data_T,coverage_period=coverage_period,name=name)

    # --- Finalisation Métadonnées ---
    processed_input.spatial_unit = df.columns.tolist()
    processed_input.C = C
    processed_input.adj_mx_path = f"{dirname}/{data_subfolder.lower()}_rn_adj.npy"
    processed_input.raw_data_path =f"{dirname}/{data_subfolder.lower()}_his_{year}.h5"
    processed_input.city = city
    # processed_input.periods = None 
    return processed_input

if __name__ == "__main__":
    # blabla
    blabla