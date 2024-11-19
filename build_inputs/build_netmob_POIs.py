import glob 
from datetime import datetime,timedelta
import os 
import sys

# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd 
import geopandas as gpd 
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

FOLDER_PATH = '../../../../data/rrochas/prediction_validation' 
POIs_path = f"{FOLDER_PATH}/POIs"
PATH_iris = f'{FOLDER_PATH}/lyon_iris_shapefile'

def get_information_from_path(path):
    day = path.split('_')[-2]
    day_str = str(day)
    day_str = datetime.strptime(day_str, '%Y%m%d')
    times = [day_str + timedelta(minutes=15*i) for i in range(96)]
    #times_str = [t.strftime('%m/%d/%Y %H:%M') for t in times]
    columns = ['tile_id'] + times # + times_str
    return(columns)

def build_netmob_ts_from_POIs(apps,stadium2tile_ids,netmob_data_FOLDER_PATH = f"{FOLDER_PATH}/../../NetMob/NetMob_raw"):
    '''
    args:
    ------
    netmob_data_FOLDER_PATH : path of the initial NetMob data 
    stadium2tile_ids :  dictonnary of (key,values) : key -> Name of POIs // values : list of NetMob tile-ids representing this POI.

    '''

    # Pour chaque 'app' et chaque mode de tranfer (UL/DL):
    for app in apps: 
        folder_path_app = f"{FOLDER_PATH}/POIs/stadium/{app}"
        if not os.path.exists(folder_path_app):
            os.mkdir(folder_path_app)
        for transfer_mode in ['DL','UL']:

            # Init DataFrames
            for key in stadium2tile_ids.keys():
                globals()[f"df_{key}"] = pd.DataFrame()

            # Load NetMob data
            folder_days = [day for day in os.listdir(f'{netmob_data_FOLDER_PATH}/{app}') if (not day.startswith('.'))] 
            for day in folder_days:      
                txt_path = glob.glob(os.path.join(f'{netmob_data_FOLDER_PATH}/{app}/{day}',f"*_{transfer_mode}.txt"))[0]
                columns = get_information_from_path(txt_path)
                df = pd.read_csv(txt_path, sep = ' ', names = columns).set_index(['tile_id'])

                # Build dataframe associated to each POIs
                for key,ids in stadium2tile_ids.items():
                    globals()[f"df_{key}"] = pd.concat([globals()[f"df_{key}"],df.loc[ids].transpose()])
            
            # Save DataFrames
            for key in stadium2tile_ids.keys():
                globals()[f"df_{key}"].sort_index().to_csv(f"{folder_path_app}/df_{key}_{transfer_mode}.csv") 
                print(f'df_{key}_{transfer_mode} saved in {folder_path_app}')
     



if __name__ == '__main__':
    from build_inputs.build_netmob_data import load_netmob_gdf
    # Load NetMob gdf
    Netmob_gdf,working_zones = load_netmob_gdf(FOLDER_PATH = FOLDER_PATH,
                                data_folder = PATH_iris, 
                                geojson_path = 'NetMob_lyon.geojson',
                                zones_path = 'lyon.shp')
    Netmob_gdf_dropped = Netmob_gdf.drop_duplicates(subset = ['tile_id'])  # Some Doubles are exis

    # Load POIs : 
    POIs = gpd.read_file(f"{POIs_path}/gdf_stadium.geojson")
    Lou_rugby = POIs[POIs.nom == 'Matmut Stadium Gerland']
    Astroballe = POIs[POIs.nom == 'Astroballe']
    Groupama = POIs[POIs.nom == 'Groupama Stadium']

    tile_ids_Lou_rugby = Netmob_gdf_dropped.sjoin(Lou_rugby)
    tile_ids_Astroballe = Netmob_gdf_dropped.sjoin(Astroballe)
    tile_ids_Groupama = Netmob_gdf_dropped.sjoin(Groupama)

    stadium2tile_ids = {'Lou_rugby':list(tile_ids_Lou_rugby.tile_id),
                        'Astroballe':list(tile_ids_Astroballe.tile_id),
                        'Groupama':list(tile_ids_Groupama.tile_id)
    }

    # Build NetMob Time-Series around POIs: 
    netmob_data_FOLDER_PATH = f"{FOLDER_PATH}/../../NetMob/NetMob_raw"
    #  Def apps : 
    #apps = [app for app in os.listdir(netmob_data_FOLDER_PATH) if ((app != 'Lyon.geojson') and (not app.startswith('.'))) ]   # Avoid hidden folder and Lyon.geojson
    apps = ['Instagram','Facebook','Uber','Google_Maps','Waze','Spotify','Deezer','Telegram','Facebook_Messenger','Snapchat','WhatsApp','Twitter', 'Pinterest']
    # ============
    build_netmob_ts_from_POIs(apps, stadium2tile_ids,netmob_data_FOLDER_PATH)

    