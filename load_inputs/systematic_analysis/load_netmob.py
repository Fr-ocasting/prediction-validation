import sys
import os
import pandas as pd
import pandas as pd 
import geopandas as gpd 
from shapely.geometry import Polygon
from datetime import datetime, timedelta
import glob
import json
current_file_path = os.path.abspath(os.getcwd())
parent_dir = os.path.abspath(os.path.join(current_file_path, '..','..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from examples.benchmark import local_get_args
from constants.paths import FOLDER_PATH
from pipeline.clustering.clustering import filter_by_temporal_agg
from load_inputs.Lyon.netmob.netmob_POIs import load_data,START, END


def load_dataset(signal,
                 NetMob_selected_apps,
                 NetMob_transfer_mode,
                 NetMob_selected_tags,
                 NetMob_expanded,
                NetMob_only_epsilon,
                 ):
    
    """
    Load the NetMob dataset for the specified signal.
    Parameters:
    signal (str): The signal to load ('NetMob_DL' or 'NetMob_UL').
    NetMob_selected_apps (list): List of selected applications.
        >> 'Google_Maps', 'Instagram',  'Deezer','Spotify',
          'Web_Weather', 'Facebook_Messenger', 'YouTube', 'WhatsApp
    NetMob_transfer_mode (list): List of transfer modes
        >> 'DL' (Download), 'UL' (Upload)
    NetMob_selected_tags (list): List of selected tags.
        >> 'park', 'stadium', 'university', 'station', 'shop',
           'nightclub', 'parkings', 'theatre', 'iris', 'transit',
           'public_transport'
    NetMob_expanded (str): Expansion option for NetMob data.
        >> '' or '_expanded'
    NetMob_only_epsilon (bool): if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'.

    
    Exemple: 
            NetMob_selected_apps= ['Google_Maps'] 
            NetMob_transfer_mode = ['DL'] 
            NetMob_selected_tags = ['iris']
            NetMob_expanded = ''
            NetMob_only_epsilon = False
    """

    invalid_dates = []
    minmaxnorm = True
    standardize = False  # Set to True if you want to standardize the data
    config = {'model_name': 'STGCN',
            'freq' : '15min',
            'dataset_names':['subway_in',signal],
            'dataset_for_coverage' : [signal],
            'target_data': 'subway_in',
                                     
            'contextual_kwargs':{signal:{'need_global_attn':True, 
                            'stacked_contextual': True,
                            'NetMob_selected_apps' : NetMob_selected_apps,
                            'NetMob_transfer_mode' :  NetMob_transfer_mode, 
                            'NetMob_selected_tags' : NetMob_selected_tags,
                            'NetMob_expanded' : NetMob_expanded, 
                            'NetMob_only_epsilon': NetMob_only_epsilon,
                            'epsilon_clustering': None, 
                            'vision_model_name' : None,
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
                tensor_limits_keeper = None
                )
    df_raw = pd.DataFrame(ds.raw_values,columns = ds.spatial_unit, index = ds.df_dates['date'])
    df =  pd.DataFrame(ds.U_train[:,:,-1].detach().cpu().numpy(),columns = ds.spatial_unit, index = ds.tensor_limits_keeper.df_verif_train.iloc[:,-2] )
    return ds, df_raw, df




def load_netmob_json(data_folder, geojson_path = 'Lyon.geojson'):
    ''' Load GeoJson, and then the spatial correspondence '''
    Lyon = json.load(open(f"{data_folder}/{geojson_path}",'r'))
    Netmob_gdf = pd.json_normalize(Lyon, record_path =['features']).rename(columns = {'properties.tile_id' : 'tile_id', 'geometry.coordinates':'geometry'})[['tile_id','geometry']]
    Netmob_gdf.geometry = Netmob_gdf.geometry.transform(lambda x : Polygon(x[0]))
    
    city_dims = {'Lyon': (426, 287)}
    city_str = 'Lyon'
    n_rows, n_cols = city_dims[city_str]
    
    return(Netmob_gdf,n_rows,n_cols)

def restrain_netmob_to_Lyon(Netmob_gdf,shp_iris_path,iris_file):
    ''' Restraint "Netmob_gdf" to the working area '''
    working_zones = gpd.read_file(f'{shp_iris_path}/{iris_file}')
    Netmob_gdf = gpd.GeoDataFrame(Netmob_gdf)
    Netmob_gdf.crs = 'epsg:4326'
    
    restrained_Lyon_gdf = gpd.sjoin(Netmob_gdf,working_zones,how = 'inner')
    return(restrained_Lyon_gdf,working_zones)


def load_netmob_gdf(folder_path,shp_iris_path,
                    geojson_path = 'Lyon.geojson',
                    iris_file = 'lyon.shp'
                    ):

    Netmob_gdf,n_rows,n_cols = load_netmob_json(folder_path, geojson_path = geojson_path)
    Netmob_gdf['centroid_lonlat'] = Netmob_gdf.geometry.apply(lambda x : x.centroid)
    Netmob_gdf = Netmob_gdf.set_geometry('centroid_lonlat')
    Netmob_gdf_joined,working_zones = restrain_netmob_to_Lyon(Netmob_gdf,shp_iris_path,iris_file) #Associate an square to an IRIS when the centroid is inside it
    
    return(Netmob_gdf_joined,working_zones,n_rows,n_cols)

def get_list_days(folder_path,app):
    list_days = [day for day in os.listdir(f'{folder_path}/{app}') if (not day.startswith('.')) ] 
    return list_days

def get_DL_UL_paths(folder_path,app,day):
    txt_paths = sorted(glob.glob(os.path.join(f'{folder_path}/{app}/{day}', "*.txt")))
    return(txt_paths)

def get_list_apps(folder_path):
    apps = [app for app in os.listdir(folder_path) if ((app != 'Lyon.geojson') and (not app.startswith('.'))) ]   # Avoid hidden folder and Lyon.geojson     
    return(apps)


def read_csv(txt_path,day):
    # let's make a list of 15 min time intervals to use as column names
    day_str = str(day)
    day = datetime.strptime(day_str, '%Y%m%d')
    times = [day + timedelta(minutes=15*i) for i in range(96)]
    times_str = [t.strftime('%H:%M') for t in times]

    # column names
    columns = ['tile_id'] + times_str

    df = pd.read_csv(txt_path, sep = ' ', names = columns).set_index(['tile_id'])
    return df