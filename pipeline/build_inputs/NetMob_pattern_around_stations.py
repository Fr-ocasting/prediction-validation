# Relative path:
import glob 
import pandas as pd
from os import listdir
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
#parent_dir = f"{parent_dir}/prediction_validation/"  # A utiliser sur .ipynb notebook
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

from pipeline.build_inputs.build_netmob_data import get_information_from_path


def read_csv(path,result,columns,dic_time_series = {},sum_on_tile_ids = True):
    df = pd.read_csv(path, sep = ' ', names = columns).set_index(['tile_id'])
    # Loop through each station. Get associated usefull Tile-id
    for station_ind in range(len(result)):
        station = result['COD_TRG'][station_ind]
        ids = result.tile_id[station_ind]
        
        df_station = df.loc[ids]
        df_station = df_station.transpose()
        df_station.index = pd.to_datetime(df_station.index)
        df_station=df_station.sort_index()
        if sum_on_tile_ids:
            df_station = df_station.sum(axis=1)
        
        if station in dic_time_series.keys():
            dic_time_series[station] = pd.concat([dic_time_series[station],df_station])
        else:
            dic_time_series[station]= df_station
    
    return(dic_time_series)

def tackle_one_day(result,netmob_data_FOLDER_PATH,app,day,dic_time_series,assert_transfer_mode):
    txt_paths = sorted(glob.glob(os.path.join(f'{netmob_data_FOLDER_PATH}/{app}/{day}', "*.txt")))
    for path in txt_paths:
        transfer_mode,columns = get_information_from_path(path)
        if transfer_mode == assert_transfer_mode:
            date_str = '-'.join([day[4:6],day[6:],day[:4]])   # date_str : 'YYYY-MM-DD' -> 'MM-DD-YYYY'
            columns = ['tile_id'] + [date_str +' '+ c for c in columns if c != 'tile_id']
            dic_time_series = read_csv(path,result,columns,dic_time_series)
    return(dic_time_series)
            
def tackle_all_days(result,netmob_data_FOLDER_PATH,folder_days,app,dic_time_series={},assert_transfer_mode='DL'):
    for k,day in enumerate(folder_days):  # day : 'YYYYMMDD'
        dic_time_series = tackle_one_day(result,netmob_data_FOLDER_PATH,app,day,dic_time_series,assert_transfer_mode)
    return(dic_time_series)

if __name__ == '__main__':
    from pipeline.build_inputs.tile_ids_around_stations import buffer_between_tile_ids_and_subway_station,netmob_data_FOLDER_PATH,PATH_iris,DATA_FOLDER_PATH
    from pipeline.build_inputs.NetMob_pattern_around_stations import tackle_all_days

    epsilon = 100
    (gdf_stations,joined,result,maxi_nb_tile) = buffer_between_tile_ids_and_subway_station(epsilon,netmob_data_FOLDER_PATH,PATH_iris)
    
    #apps = [app for app in listdir(netmob_data_FOLDER_PATH) if ((app != 'Lyon.geojson') and (not app.startswith('.'))) ]   # Avoid hidden folder and Lyon.geojson
    apps = ['Google_Maps','Uber','Facebook','Instagram','WhatsApp','Deezer']
    for app in apps: 
        print('App: ',app)
        folder_days = [day for day in listdir(f'{netmob_data_FOLDER_PATH}/{app}') if (not day.startswith('.')) ] 
        dic_time_series = tackle_all_days(result,netmob_data_FOLDER_PATH,folder_days,app,dic_time_series={},assert_transfer_mode='DL')
        break