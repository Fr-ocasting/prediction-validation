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
 - add argument 'n_vertex', 'C' to the NameSpace. These are specific to this data
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''

FILE_NAME = 'subway_out/subway_out'  # 'subway_out/subway_out'  #  'subway_in/subway_in' 

list_of_invalid_period = []
list_of_invalid_period.append([datetime(2019,1,10,15,30),datetime(2019,1,14,15,30)])
list_of_invalid_period.append([datetime(2019,1,30,8,15),datetime(2019,1,30,10,30)])
list_of_invalid_period.append([datetime(2019,2,18,11),datetime(2019,2,18,13)])
list_of_invalid_period.append([datetime(2019,4,23,14),datetime(2019,4,28,14)])
list_of_invalid_period.append([datetime(2019,6,26,11),datetime(2019,6,28,4)])
list_of_invalid_period.append([datetime(2019,10,27),datetime(2019,10,28,16)])
list_of_invalid_period.append([datetime(2019,12,21,15,45),datetime(2019,12,21,16,45)])

INVALID_DATES = []
for start,end in list_of_invalid_period:
    INVALID_DATES = INVALID_DATES + list(pd.date_range(start,end,freq = f'15min'))
C = 1
n_vertex = 40
COVERAGE = pd.date_range(start='03/16/2019', end='06/01/2019', freq='15min')[:-1]


def load_data(dataset,args,ROOT,FOLDER_PATH,intesect_coverage_period,normalize,invalid_dates):
    id_stations = dataset.spatial_unit
    contextual_subway_out = []

    dims = [0]
    subway_out = load_data_from_subway_in_py(args,ROOT,FOLDER_PATH,intesect_coverage_period)
    for id_station in id_stations:
        print('Spatial unit: ',id_station)

        netmob_T = torch.Tensor(subway_out.raw_values)
        #print(">>>>> ICI ON UTILISE LE SUBWAY IN FUTURE !!!!")
        #print(">>>>> CAS NORMAL IL FAUT CHANGER 'FILE_NAME' ET COMMENTER LE ROLL ET DE-COMMENTER LA LIGNE DESSUS !!!!")
        #netmob_T = torch.roll(torch.Tensor(subway_out.raw_values), shifts=-1, dims=0)

        preprocessed_personal_input = load_input_and_preprocess(dims = dims,normalize=normalize,invalid_dates=invalid_dates,args=args,netmob_T=netmob_T,dataset=dataset)
        preprocessed_personal_input.station_name = id_station
        contextual_subway_out.append(preprocessed_personal_input)
    return contextual_subway_out

