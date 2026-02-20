import sys
import os
import pandas as pd
import geopandas as gpd 
current_file_path = os.path.abspath(os.getcwd())
parent_dir = os.path.abspath(os.path.join(current_file_path, '..','..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from constants.config import local_get_args
from constants.paths import FOLDER_PATH



def load_dataset(signal,agg_iris_target_n=None,threshold_volume_min=1,):
    if signal == 'bike_in' : 
        from load_inputs.Lyon.bike.bike_in import load_data, START, END
    if signal == 'bike_out' :
        from load_inputs.Lyon.bike.bike_out import load_data, START, END


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
            'target_kwargs':{signal:{'agg_iris_target_n':agg_iris_target_n,
                                     'threshold_volume_min':threshold_volume_min,
                                    },
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
                tensor_limits_keeper = None,
                )
    df_raw = pd.DataFrame(ds.raw_values,columns = ds.spatial_unit, index = ds.df_dates['date'])
    df =  pd.DataFrame(ds.U_train[:,:,-1].detach().cpu().numpy(),columns = ds.spatial_unit, index = ds.tensor_limits_keeper.df_verif_train.iloc[:,-2] )
    return ds, df_raw, df


def load_iris(ds_in,agg_iris_target_n):
    # Load Iris
    remaining_iris_zones = ds_in.spatial_unit
    if agg_iris_target_n is None:
        gdf_iris = gpd.read_file(os.path.join(FOLDER_PATH, f'lyon_iris_shapefile', 'lyon.shp'))
    else:
        gdf_iris = gpd.read_file(os.path.join(FOLDER_PATH, f'lyon_iris_agg{agg_iris_target_n}', 'lyon.shp'))
    gdf_iris['remaining_iris_zones'] = gdf_iris.index.isin(remaining_iris_zones)
    gdf_iris['spatial_unit'] = gdf_iris.index
    return gdf_iris

def load_bike_gdf(inflow,outflow):
    # load bike sharing stations : 
    json_path = 'pvo_patrimoine_voirie.pvostationvelov.json'
    gdf_velov = gpd.read_file(os.path.join(FOLDER_PATH, json_path))

    gdf_velov = pd.merge(gdf_velov, inflow, on='idstation')
    gdf_velov = pd.merge(gdf_velov, outflow, on='idstation')
    return gdf_velov
