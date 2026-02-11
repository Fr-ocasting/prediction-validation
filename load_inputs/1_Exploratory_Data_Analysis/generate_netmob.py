import datetime 

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import bokeh
import folium 
import geopandas as gpd 
current_file_path = os.path.abspath(os.getcwd())
parent_dir = os.path.abspath(os.path.join(current_file_path, '..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from constants.paths import FOLDER_PATH
from load_inputs.systematic_analysis.load_netmob import load_dataset,get_list_apps,get_list_days,get_DL_UL_paths,read_csv,load_netmob_gdf
from load_inputs.systematic_analysis.utils import get_histogram_per_day_type,get_boxplot_per_spatial_unit_per_day_type,heatmap
from load_inputs.systematic_analysis.utils import IN_bdc,OUT_bdc,IN_nbdc,OUT_nbdc

folder_path = f'{FOLDER_PATH}/../raw_data/NetMob/'
shp_iris_path = f'{FOLDER_PATH}/../raw_data/lyon_iris_shapefile/'
path_save = '../../../../../../data/NetMob/NetMob_Lyon'
folder_path = '../../../../../../data/NetMob/NetMob_raw'

Netmob_gdf_joined,working_zones,n_rows,n_cols = load_netmob_gdf(folder_path,shp_iris_path)
Netmob_gdf_joined = Netmob_gdf_joined.drop_duplicates('tile_id')
Netmob_gdf_joined = Netmob_gdf_joined[['tile_id','INSEE_COM','NOM_COM','NOM_IRIS','geometry']]
Netmob_gdf_joined = gpd.GeoDataFrame(Netmob_gdf_joined,crs='EPSG:4326') #Don't know why, but the geodataframe seem corrupted as we can't convert it into GeoJson, that's why we need to use "gpd.GeoDataFrame()"
Netmob_gdf_joined
Netmob_gdf_joined.head()

app_i = 2
day_i = 0

apps = get_list_apps(folder_path)
print(apps)

app = 'Google_Maps' 

""" Choices: 

['Apple_Video', 'Google_Play_Store', 'Google_Maps', 'Web_Clothes', 'Uber', 'Twitter', 'Microsoft_Mail', 'Microsoft_Store', 'Apple_Music', 'Microsoft_Office', 'Pokemon_GO', 'Clash_of_Clans', 'Yahoo_Mail', 'PlayStation', 'Wikipedia', 'Apple_Web_Services', 'Pinterest', 'Web_Ads', 'Google_Mail', 'Google_Meet', 'Apple_Siri', 'Web_Adult', 'Spotify', 'Deezer', 'Waze', 'Web_Games', 'Apple_App_Store', 'Microsoft_Skydrive', 'Google_Docs', 'Microsoft_Web_Services', 'Molotov', 'YouTube', 'Apple_iTunes', 'Apple_iMessage', 'DailyMotion', 'Netflix', 'Web_Transportation', 'Web_Downloads', 'SoundCloud', 'TeamViewer', 'Google_Web_Services', 'Facebook', 'EA_Games', 'Tor', 'Amazon_Web_Services', 'Web_e-Commerce', 'Telegram', 'Apple_Mail', 'Dropbox', 'Web_Food', 'Apple_iCloud', 'Skype', 'Facebook_Messenger', 'Twitch', 'Microsoft_Azure', 'Instagram', 'Facebook_Live', 'Web_Streaming', 'Orange_TV', 'Periscope', 'Snapchat', 'Web_Finance', 'WhatsApp', 'Web_Weather', 'Google_Drive', 'LinkedIn', 'Yahoo', 'Fortnite']
"""

for app in apps:
    print(app)
    # Get csv paths: 
    list_days = get_list_days(folder_path,app)

    df_tot = pd.DataFrame()

    for k,day in enumerate(list_days):
        year = day[:4]
        month = day[4:6]
        day_day = day[6:8]
        txt_paths = get_DL_UL_paths(folder_path,app,day)
        txt_path_DL = txt_paths[0]

        # Load df
        df = read_csv(txt_path_DL,day).T
        df.index = pd.to_datetime([f"{year}-{month}-{day_day} {t}" for t in df.index],format="%Y-%m-%d %H:%M")
        df_tot = pd.concat([df_tot,df])
        

    # Restrict NetMob to Lyon: 
    df = df_tot.sort_index()[list(Netmob_gdf_joined['tile_id'])]



    # Save Dataset: 
    np.save(f'{path_save}/{app}.npy', df.values)
    pd.Series(df.index).to_pickle(f'{path_save}/index.pkl')
    pd.Series(df.columns).to_pickle(f'{path_save}/columns.pkl')