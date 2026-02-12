import sys 
import os 
import pandas as pd
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from datetime import datetime 
import torch
from load_inputs.Lyon.pt.subway_in import load_DataSet 
from pipeline.DataSet.dataset import DataSet
from pipeline.build_inputs.load_preprocessed_dataset import load_input_and_preprocess
from pipeline.utils.utilities import filter_args
''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - add argument 'num_nodes', 'C' to the NameSpace. These are specific to this data
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''
NAME = 'subway_in_subway_out'

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
# list_of_invalid_period.append([datetime(2019,12,21,15,45),datetime(2019,12,21,16,45)])


C = 2
#num_nodes = 40

def load_data(FOLDER_PATH,invalid_dates,coverage_period,args,minmaxnorm,standardize,normalize,tensor_limits_keeper = None):
    data_T_list = []
    spatial_units = None
    indices_spatial_unit = None
    for name in ['subway_in','subway_out']:
        dataset = load_DataSet(args,FOLDER_PATH,coverage_period = coverage_period,filename=f"{name}/{name}",name=name)

        if spatial_units is None:
            spatial_units = dataset.spatial_unit
            indices_spatial_unit = dataset.indices_spatial_unit
        else:
            assert (spatial_units == dataset.spatial_unit).all() and (indices_spatial_unit == dataset.indices_spatial_unit).all(), "The spatial units of the different datasets are not the same !"


        if  hasattr(args,'contextual_kwargs') and (name in args.contextual_kwargs.keys()) and ('use_future_values' in args.contextual_kwargs[name].keys()) and args.contextual_kwargs[name]['use_future_values'] and ('loading_contextual_data' in args.contextual_kwargs[name].keys()) and args.contextual_kwargs[name]['loading_contextual_data']:
            data_T = torch.roll(torch.Tensor(dataset.raw_values), shifts=-1, dims=0)
            print(f">>>>> ICI ON UTILISE LE {name.upper()} IN FUTURE !!!!")
            print('data_T.size: ',data_T.size())
        else:
            data_T = dataset.raw_values
        data_T_list.append(data_T)
    data_T = torch.stack(data_T_list,dim=1)
        

    preprocesed_ds = load_input_and_preprocess(dims = dataset.dims,normalize=normalize,invalid_dates=invalid_dates,args=args,data_T=data_T,
                                            coverage_period=coverage_period,name=name,
                                            minmaxnorm=minmaxnorm,standardize=standardize,
                                            tensor_limits_keeper=tensor_limits_keeper)
    
    preprocesed_ds.spatial_unit = dataset.spatial_unit
    preprocesed_ds.dims = dataset.dims
    preprocesed_ds.periods = dataset.periods
    preprocesed_ds.time_step_per_hour = dataset.time_step_per_hour
    preprocesed_ds.indices_spatial_unit = dataset.indices_spatial_unit
    preprocesed_ds.city = dataset.city
    return preprocesed_ds

