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
from load_inputs.PeMS03 import load_data as load_data_from_PEMS03

"""
PEMS04 Dataset
df.shape:  (16992, 307)
from https://github.com/RWLinno/ST-LoRA/data
"""
FILE_BASE_NAME = '4'
YEAR = 2018
DATA_SUBFOLDER = f'PEMS0{FILE_BASE_NAME}' 
CITY = f'California_{DATA_SUBFOLDER}'

# Naive Freq
NATIVE_FREQ = '5min'
# Temporal Coverage period
START = '2018-01-01 00:00:00'
END = '2018-03-01 00:00:00'
USELESS_DATES = {'hour':[], #[1,2,3,4,5,6],  #[] if no useless (i.e removed) hours
                 'weekday':[]#[5,6],
                 }
# List of invalid period 
list_of_invalid_period = []

C = 1 # Nb channels by spatial units

def load_data(FOLDER_PATH, invalid_dates, coverage_period, args, normalize=True):
    return load_data_from_PEMS03(FOLDER_PATH, invalid_dates, coverage_period, args, normalize=True,
              data_subfolder = DATA_SUBFOLDER,
            year = YEAR,
            city = CITY)


if __name__ == "__main__":
    # blabla
    blabla