import sys
import os

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Personnal import
from load_inputs.PeMS08 import load_data as load_data_from_PEMS08

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
CHANNELS = 'speed' # 1 channel


def load_data(FOLDER_PATH, invalid_dates, coverage_period, args, normalize=True):
    return load_data_from_PEMS08(FOLDER_PATH, invalid_dates, coverage_period, args, normalize=True,
              data_subfolder = DATA_SUBFOLDER,
            year = YEAR,
            city = CITY,
            name=NAME,
            channel_name=CHANNELS)


if __name__ == "__main__":
    # blabla
    blabla