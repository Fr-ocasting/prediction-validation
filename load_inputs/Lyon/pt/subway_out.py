import sys 
import os 
import pandas as pd
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from datetime import datetime 
import torch
from load_inputs.Lyon.pt.subway_in import load_data as load_data_from_subway_in_py
from pipeline.build_inputs.load_preprocessed_dataset import load_input_and_preprocess
from pipeline.utils.utilities import filter_args
''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - add argument 'num_nodes', 'C' to the NameSpace. These are specific to this data
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''
NAME = 'subway_out'
FILE_NAME = 'subway_out/subway_out'  # 'subway_out/subway_out'  #  'subway_in/subway_in' 
# START = '03/16/2019'
# END = '06/01/2019'
# START = '01/01/2019'
# END = '10/23/2020'
START = '01/01/2019'
END = '01/01/2020'

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
list_of_invalid_period.append([datetime(2019,10,27),datetime(2019,10,28,20,15)])
list_of_invalid_period.append([datetime(2019,12,21,15,45),datetime(2019,12,21,16,45)])


C = 1
#num_nodes = 40

def load_data(FOLDER_PATH,invalid_dates,coverage_period,args,minmaxnorm,standardize,normalize,tensor_limits_keeper = None):
    
    preprocesed_ds = load_data_from_subway_in_py(FOLDER_PATH,
                                            invalid_dates = invalid_dates,
                                            coverage_period = coverage_period,
                                            args = args,
                                            minmaxnorm = minmaxnorm,
                                            standardize = standardize,
                                            normalize= normalize,
                                            filename = FILE_NAME,
                                            name=NAME,
                                            tensor_limits_keeper = tensor_limits_keeper)

    # Si on souhaite utiliser le subway-in future, il suffit de dé-commenter les lignes en dessous, et changer le FILE_NAME:
    # print("\n>>>>> ICI ON UTILISE LE SUBWAY IN FUTURE !!!!")
    # T_subway_out = torch.Tensor(subway_out.raw_values.float())
    # dims = [0]
    # netmob_T = torch.roll(torch.Tensor(subway_out.raw_values), shifts=-1, dims=0)
    
    # preprocesed_ds = load_input_and_preprocess(dims = dims,normalize=normalize,invalid_dates=invalid_dates,args=args,data_T=T_subway_out,
    #                                                         coverage_period=coverage_period,name=NAME,
    #                                                         minmaxnorm=minmaxnorm,standardize=standardize,
    #                                                         tensor_limits_keeper=tensor_limits_keeper
    #                                                           )
    # preprocesed_ds.periods = subway_out.periods
    # preprocesed_ds.spatial_unit = subway_out.spatial_unit
    # preprocesed_ds.C = C
    return preprocesed_ds

