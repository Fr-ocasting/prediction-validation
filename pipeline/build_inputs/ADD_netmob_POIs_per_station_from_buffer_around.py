# GET PARAMETERS
import os 
from os import listdir
import sys
import pandas as pd 
import numpy as np 
import pickle
# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from pipeline.build_inputs.tile_ids_around_stations import DATA_FOLDER_PATH
from pipeline.build_inputs.tile_ids_around_stations import buffer_between_tile_ids_and_subway_station,netmob_data_FOLDER_PATH,PATH_iris
from pipeline.build_inputs.NetMob_pattern_around_stations import tackle_all_days
from pipeline.load_inputs.subway_in import get_trigram_correspondance


ROOT = f"{DATA_FOLDER_PATH}/rrochas"
FOLDER_PATH = 'prediction_validation'
subway_stations = get_trigram_correspondance()

apps = ['Apple_Video','Google_Play_Store','Google_Maps','Web_Clothes','Uber', 'Twitter',
 'Microsoft_Mail', 'Microsoft_Store', 'Apple_Music', 'Microsoft_Office', 'Pokemon_GO', 'Clash_of_Clans', 'Yahoo_Mail', 'PlayStation',
'Wikipedia', 'Apple_Web_Services', 'Pinterest', 'Web_Ads', 'Google_Mail', 'Google_Meet',
 'Apple_Siri', 'Web_Adult', 'Spotify', 'Deezer', 'Waze', 'Web_Games', 'Apple_App_Store', 'Microsoft_Skydrive', 'Google_Docs', 'Microsoft_Web_Services',
 'Molotov', 'YouTube', 'Apple_iTunes', 'Apple_iMessage', 'DailyMotion', 'Netflix', 'Web_Transportation',
 'Web_Downloads', 'SoundCloud', 'TeamViewer', 'Google_Web_Services', 'Facebook', 'EA_Games', 'Tor', 'Amazon_Web_Services',
 'Web_e-Commerce', 'Telegram', 'Apple_Mail','Dropbox', 'Web_Food', 'Apple_iCloud', 'Skype', 'Facebook_Messenger', 'Twitch', 'Microsoft_Azure',
   'Instagram', 'Facebook_Live', 'Web_Streaming', 'Orange_TV', 'Periscope', 'Snapchat' ,'Web_Finance' ,'WhatsApp', 'Web_Weather','Google_Drive','LinkedIn','Yahoo','Fortnite']


if False : 
    apps_to_filter = ['Instagram','Facebook','Uber','Google_Maps','Waze','Spotify','Deezer','Telegram','Facebook_Messenger','Snapchat','WhatsApp','Twitter','Pinterest']
    apps = [x for x in apps if not x in apps_to_filter]


list_COD_TRG = list(subway_stations.COD_TRG)
epsilon_list = [100,300] # [100,200,300]

## __ Load tile_ids within buffer : 
for epsilon in epsilon_list: 
    _,_,globals()[f"result_epsilon{epsilon}"],_ = buffer_between_tile_ids_and_subway_station(epsilon,netmob_data_FOLDER_PATH,PATH_iris)


## __ Generate the new data from buffer :
ts_stations_epsilons_transfers_apps = []
for app in apps:
    list_transfer = []
    for transfer_mode in ['DL','UL']:
        print(f"App: {app}_{transfer_mode}")
        list_epsilon = []
        for epsilon in epsilon_list:
            folder_days = [day for day in listdir(f'{netmob_data_FOLDER_PATH}/{app}') if (not day.startswith('.')) ] 
            dic_time_series = tackle_all_days(globals()[f"result_epsilon{epsilon}"],netmob_data_FOLDER_PATH,folder_days,app,dic_time_series={},assert_transfer_mode=transfer_mode)
            ts_stations_app_r_epsilon_k_transfer_p = np.array([dic_time_series[station_i].sort_index() for station_i in list_COD_TRG])
            list_epsilon.append(ts_stations_app_r_epsilon_k_transfer_p)
        list_transfer.append(np.array(list_epsilon))
    ts_stations_epsilons_transfers_apps.append(np.array(list_transfer))
ts_stations_epsilons_transfers_apps = np.array(ts_stations_epsilons_transfers_apps)
# ...

# [Apps,Transfer_mode,Osmid,Stations,T]  -> [Stations,Apps,Osmid,Transfer_mode,T]
ts_stations_epsilons_transfers_apps = ts_stations_epsilons_transfers_apps.transpose(3,0,2,1,4)

print('All Data Generated')

""" Create a new 'InputsEpsilon' with not any POIs. Only app data distant from an epsilon to the station """
if False: 
    for NetMob_expanded in ['','_expanded']:
        save_folder = f"{ROOT}/{FOLDER_PATH}/POIs/netmob_POI_Lyon{NetMob_expanded}/InputsEpsilon/"
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
    ## __ Save the New Added Buffer : 
    for ind_station,station_i in enumerate(list_COD_TRG):
        print('Start save ',station_i)
        for NetMob_expanded in ['','_expanded']:
            save_folder = f"{ROOT}/{FOLDER_PATH}/POIs/netmob_POI_Lyon{NetMob_expanded}/InputsEpsilon/{station_i}"
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            new_data = ts_stations_epsilons_transfers_apps[ind_station]   #[Apps,Osmid,Transfer_mode,T]

            metadata = {'apps':apps,  # list of app >>> ['Instagram','Google_Maps']
                        'id_stations':station_i,  # station name >>> 'AMP'
                        'expanded': NetMob_expanded, # '' or '_expanded'
                        'transfer_modes': [], # list of transfer modes >>> ['DL','UL']
                        'shape': '[len(apps),len(osmid),len(transfer_modes),len(dates)]',
                        'osmid': [f'station_epsilon{epsilon}'for epsilon in epsilon_list],  # list of osmid. could be 'AMP','station_epsilon300', or a real osmid
                        'tags': [f'station_epsilon{epsilon}'for epsilon in epsilon_list],  # list of osmid. could be 'station', 'station_epsilon300', or related to real tag like 'university', or 'park'
                        'dates': []# Index(['2019-03-16 00:00:00', '2019-03-16 00:15:00', '2019-03-16 00:30:00', '2019-03-16 00:45:00' ....])
                        }
            
            save_folder_ref = f"{ROOT}/{FOLDER_PATH}/POIs/netmob_POI_Lyon{NetMob_expanded}/Inputs/{station_i}"
            metadata_ref = pickle.load(open(f"{save_folder_ref}/metadata.pkl","rb"))

            metadata['dates'] = metadata_ref['dates']
            metadata['transfer_modes'] = metadata_ref['transfer_modes']

            with open(f"{save_folder}/data.npy", 'wb') as f_data:
                np.save(f_data,new_data)
            with open(f"{save_folder}/metadata.pkl", 'wb') as f_meta:
                pickle.dump(metadata,f_meta)


"""Add new data on 'agg_TS': """
if False:
    import numpy as np 
    import pickle
    from pipeline.load_inputs.subway_in import get_trigram_correspondance
    stations = list(get_trigram_correspondance().COD_TRG)
    root = '/home/rrochas/prediction-validation/../../../../data/rrochas/prediction_validation/POIs/netmob_POI_Lyon'
    save_folder = f'{root}/InputsEpsilon'

    # Init : 
    save_folder_i = f"{save_folder}/AMP"
    metadata = pickle.load(open(f"{save_folder_i}/metadata.pkl","rb"))
    for osmid_i,osmid in enumerate(metadata['osmid']):
        for transfer_mode_i,transfer_mode in enumerate(metadata['transfer_modes']):
            for app_i,app in enumerate(metadata['apps']):
                globals()[f"data_{osmid}_{app}_{transfer_mode}"] = []
    # ...

    # For each station, transfer mode, app ...:
    for station in stations:
        save_folder_i = f"{save_folder}/{station}"
        metadata = pickle.load(open(f"{save_folder_i}/metadata.pkl","rb"))
        data = np.load(open(f"{save_folder_i}/data.npy",'rb'))

        for osmid_i,osmid in enumerate(metadata['osmid']):
            for transfer_mode_i,transfer_mode in enumerate(metadata['transfer_modes']):
                for app_i,app in enumerate(metadata['apps']):
                    data_station_i = data[app_i,osmid_i,transfer_mode_i,:]
                    globals()[f"data_{osmid}_{app}_{transfer_mode}"].append(data_station_i)


    new_save_folder = f'{root}/Inputs/agg_TS'
    for osmid_i,osmid in enumerate(metadata['osmid']):
        for transfer_mode_i,transfer_mode in enumerate(metadata['transfer_modes']):
            for app_i,app in enumerate(metadata['apps']):
                globals()[f"data_{osmid}_{app}_{transfer_mode}"] = np.stack(globals()[f"data_{osmid}_{app}_{transfer_mode}"])
                new_save_path = f"{new_save_folder}/{osmid}/{app}/{transfer_mode}"
                if not os.path.exists(new_save_path):
                    os.makedirs(new_save_path)
                    with open(f"{new_save_path}/data.npy", 'wb') as f_data:
                        np.save(f_data,globals()[f"data_{osmid}_{app}_{transfer_mode}"])



""" Add to 'Inputs' with some POIs already extracted for specifics apps"""
if False:
    ## __ Save the New Added Buffer : 
    for ind_station,station_i in enumerate(list_COD_TRG):
        print('Start save ',station_i)
        for NetMob_expanded in ['','_expanded']:
            save_folder = f"{ROOT}/{FOLDER_PATH}/POIs/netmob_POI_Lyon{NetMob_expanded}/Inputs/{station_i}"
            data_app = np.load(open(f"{save_folder}/data.npy","rb"))
            metadata = pickle.load(open(f"{save_folder}/metadata.pkl","rb"))
            data_to_add = ts_stations_epsilons_transfers_apps[ind_station]   #[Apps,Osmid,Transfer_mode,T]
            new_concatenated_data = np.concatenate([data_app,data_to_add],1)

            for epsilon in epsilon_list:
                name_ts = f"station_epsilon{epsilon}"
                metadata['osmid'].append(name_ts)
                metadata['tags'].append(name_ts)

            with open(f"{save_folder}/data.npy", 'wb') as f_data:
                np.save(f_data,new_concatenated_data)
            with open(f"{save_folder}/metadata.pkl", 'wb') as f_meta:
                pickle.dump(metadata,f_meta)
    # ...