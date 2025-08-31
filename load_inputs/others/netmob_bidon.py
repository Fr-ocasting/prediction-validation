import sys 
import os 
import pandas as pd
import torch 

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
from datetime import datetime 
from pipeline.dataset import DataSet
from pipeline.utils.utilities import get_time_step_per_hour
''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - add argument 'num_nodes', 'C' to the NameSpace. These are specific to this data
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''
NAME = 'netmob_bidon'
FILE_NAME = 'netmob_bidon'
START = '03/16/2019'
END = '06/01/2019'
USELESS_DATES = {'hour':[], #[1,2,3,4,5,6],  #[] if no useless (i.e removed) hours
                 'weekday':[]#[5,6],
                 }
FREQ = '15min'

list_of_invalid_period = []
list_of_invalid_period.append([datetime(2019,5,16,0,0),datetime(2019,5,16,18,15)])  # 16 mai 00:00 - 18:15
list_of_invalid_period.append([datetime(2019,5,11,23,15),datetime(2019,5,12,0,0)])  # 11 mai 23:15 - 11 mai 23:59: down META (fb, whatsapp)
list_of_invalid_period.append([datetime(2019,5,23,0,0),datetime(2019,5,25,6,0)])  # Anoamlies for every single apps  23-25 May


C = 1
num_nodes = 10


def load_data(args,FOLDER_PATH,coverage_period = None):
    '''Load the dataset. Supposed to coontains pd.DateTime Index as index, and named columns.
    columns has to represent the spatial units.

    outputs: 
    ---------
    df: contains 
    df.index : coverage period of the dataset 
    invalid_dates : list of invalid dates 
    '''

    tensor = torch.load(f"{FOLDER_PATH}/{FILE_NAME}.pt")
    dates = pd.date_range(start=START, end=END, freq=args.freq)[:1000]

    tensor = restrain_tensor_to_specific_period(tensor,dates,coverage_period)
    time_step_per_hour = get_time_step_per_hour(args.freq)
    if args.freq != FREQ :
        assert int(args.freq.replace('min',''))> int(FREQ.replace('min','')), f'Trying to apply a a {args.freq} temporal aggregation while the minimal possible one is {FREQ}'
        tensor = tensor.view(-1, int(args.freq.replace('min','')) // int(FREQ.replace('min','')), *tensor.shape[1:]).sum(dim=1)

    dataset = DataSet(tensor = tensor,
                      dates = dates,
                      time_step_per_hour=time_step_per_hour, 
                      Weeks = args.W, 
                      Days = args.D, 
                      historical_len= args.H,
                      step_ahead=args.step_ahead,
                      data_augmentation= args.data_augmentation
                      )
    
    raise NotImplementedError('DEVRAIT ETRE PersonnalInput.preprocess')
    return(dataset)
    
def restrain_tensor_to_specific_period(tensor,dates,coverage_period):
    if coverage_period is not None:
        coverage_indices = [k for k,date in dates if date in coverage_period]
        tensor = tensor[coverage_indices]
    return tensor