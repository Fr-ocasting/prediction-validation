import sys 
import os 
import pandas as pd
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from datetime import datetime 
import torch
from load_inputs.subway_in import load_data as load_data_from_subway_in_py
from load_inputs.netmob_POIs import load_input_and_preprocess
from utils.utilities import filter_args
''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - add argument 'num_nodes', 'C' to the NameSpace. These are specific to this data
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''
NAME = 'subway_out_per_station'
FILE_NAME = 'subway_out/subway_out'  # 'subway_out/subway_out'  #  'subway_in/subway_in' 
START = '03/16/2019'
END = '06/01/2019'
FREQ = '15min'
USELESS_DATES = {'hour':[1,2,3,4,5,6],  #[] if no useless (i.e removed) hours
                 'weekday':[]#[5,6],
                 }
list_of_invalid_period = []
list_of_invalid_period.append([datetime(2019,1,10,15,30),datetime(2019,1,14,15,30)])
list_of_invalid_period.append([datetime(2019,1,30,8,15),datetime(2019,1,30,10,30)])
list_of_invalid_period.append([datetime(2019,2,18,11),datetime(2019,2,18,13)])
list_of_invalid_period.append([datetime(2019,4,23,14),datetime(2019,4,28,14)])
list_of_invalid_period.append([datetime(2019,6,26,11),datetime(2019,6,28,4)])
list_of_invalid_period.append([datetime(2019,10,27),datetime(2019,10,28,16)])
list_of_invalid_period.append([datetime(2019,12,21,15,45),datetime(2019,12,21,16,45)])


C = 1
num_nodes = 40

def load_data(dataset,args,FOLDER_PATH,intersect_coverage_period,minmaxnorm,standardize,normalize,invalid_dates):
    id_stations = dataset.spatial_unit
    contextual_subway_out = []

    dims = [0]
    subway_out = load_data_from_subway_in_py(args,FOLDER_PATH,intersect_coverage_period,filename = FILE_NAME)
    for id_station in id_stations:
        T_subway_out = torch.Tensor(subway_out.raw_values)

        # Si on souhaite utiliser le subway-in future, il suffit de dé-commenter les trois lignes en dessous, et changer le FILE_NAME:
        #print("\n>>>>> ICI ON UTILISE LE SUBWAY IN FUTURE !!!!")
        #netmob_T = torch.roll(torch.Tensor(subway_out.raw_values), shifts=-1, dims=0)

        preprocessed_personal_input = load_input_and_preprocess(dims = dims,normalize=normalize,invalid_dates=invalid_dates,args=args,netmob_T=T_subway_out,dataset=dataset,name=NAME,
                                                                minmaxnorm=minmaxnorm,standardize=standardize)
        preprocessed_personal_input.station_name = id_station
        preprocessed_personal_input.periods = subway_out.periods
        preprocessed_personal_input.spatial_unit = subway_out.spatial_unit
        contextual_subway_out.append(preprocessed_personal_input)
    return contextual_subway_out
