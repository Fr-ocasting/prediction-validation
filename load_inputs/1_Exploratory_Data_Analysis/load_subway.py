import sys
import os
import pandas as pd
current_file_path = os.path.abspath(os.getcwd())
parent_dir = os.path.abspath(os.path.join(current_file_path, '..','..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from examples.benchmark import local_get_args
from constants.paths import FOLDER_PATH
from pipeline.clustering.clustering import filter_by_temporal_agg



def load_dataset(signal):
    if signal == 'subway_in' : 
        from load_inputs.Lyon.pt.subway_in import load_data, START, END,FILE_NAME
    if signal == 'subway_out' :
        from load_inputs.Lyon.pt.subway_out import load_data,START, END,FILE_NAME


    invalid_dates = []
    minmaxnorm = True
    standardize = False  # Set to True if you want to standardize the data
    config = {'model_name': 'STGCN',
            'freq' : '15min',
            'dataset_names':[signal],
            'dataset_for_coverage' : [signal],
            'target_data': signal,
            'contextual_kwargs':{signal:{},
                            },
            'step_ahead':1,
            'horizon_step':1,
            }

    args = local_get_args(config['model_name'],
                    args_init = None,
                    dataset_names=config['dataset_names'],
                    dataset_for_coverage=config['dataset_for_coverage'],
                    modification = config)


    coverage_period = pd.date_range(start=START, end=END, freq='15min')[:-1]  # Exclude the last date to match the data



    ds = load_data(FOLDER_PATH, 
                coverage_period=coverage_period, 
                invalid_dates=invalid_dates, 
                args=args, minmaxnorm=minmaxnorm,
                standardize=standardize, 
                normalize=True,
                filename=FILE_NAME,
                tensor_limits_keeper = None
                )
    df_raw = pd.DataFrame(ds.raw_values,columns = ds.spatial_unit, index = ds.df_dates['date'])
    df =  pd.DataFrame(ds.U_train[:,:,-1].detach().cpu().numpy(),columns = ds.spatial_unit, index = ds.tensor_limits_keeper.df_verif_train.iloc[:,-2] )
    return ds, df_raw, df
