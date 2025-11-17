import os 
import sys
import torch 

# Get Parent folder : 
current_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

# Usual paths: 
if torch.cuda.is_available():
    FOLDER_PATH = os.path.expanduser('~/../../data/rrochas/prediction_validation')
    ROOT = os.path.expanduser('~/../../home/rrochas/prediction-validation')
    # Load the target data to predict: 
    #DATA_TO_PREDICT = 'subway_indiv' # 'subway_indiv', 'data_bidon'  # 'subway_in' # 'METR_LA' # 'PEMS_BAY' #'CRITER_3lanes'
    ABS_PATH_PACKAGE = '/home/rrochas/prediction-validation'
    #FILE_NAME = 'preprocessed_subway_15_min' #.csv
    #FILE_NAME = 'subway_IN_interpol_neg_15_min_16Mar2019_1Jun2020' #.csv
    #FILE_NAME = 'subway_IN_interpol_neg_15_min_2019_2020' #.csv
else:
    #current_path = os.path.abspath(os.path.dirname(__file__))
    #package_path = os.path.abspath(os.path.join(current_path, '..'))
    #FOLDER_PATH = '../../../../Data/'
    FOLDER_PATH =  '../data'
    ROOT = ''
    #DATA_TO_PREDICT = 'subway_in' # 'data_bidon'  # 'subway_in' # 'METR_LA' # 'PEMS_BAY' # 'CRITER_3lanes'
    ABS_PATH_PACKAGE = '/Users/rrochas/Desktop/Code/prediction-validation'
    #FILE_NAME = 'data_bidon' #.csv
    
# SAVE_DIRECTORY = os.path.expanduser('~/../../home/rrochas/prediction-validation/save')    
SAVE_DIRECTORY = os.path.expanduser(f'{parent_dir}/save')


DATASET_IMPORT_PATHS = {
    # --- Lyon : 
    # PT: 
    'subway_in': 'load_inputs.Lyon.pt.subway_in',
    'subway_out': 'load_inputs.Lyon.pt.subway_out',
    'subway_indiv': 'load_inputs.Lyon.pt.subway_indiv',
    'subway_out_per_station': 'load_inputs.Lyon.pt.subway_out_per_station',
    'subway_in_subway_out': 'load_inputs.Lyon.pt.subway_in_subway_out',
    'tramway_indiv': 'load_inputs.Lyon.pt.tramway_indiv',
    'buses_indiv': 'load_inputs.Lyon.pt.buses_indiv',

    # Bike 
    'bike_in': 'load_inputs.Lyon.bike.bike_in',
    'bike_out': 'load_inputs.Lyon.bike.bike_out',

    # Car
    'CRITER_3lanes': 'load_inputs.car.CRITER_3lanes',
    'CRITER_3_4_5_lanes_flow': 'load_inputs.car.CRITER_3_4_5_lanes_flow',
    'CRITER_3_4_5_lanes_occupancy': 'load_inputs.car.CRITER_3_4_5_lanes_occupancy',
    'CRITER_sup_35000dailyflow_3_4_5_lanes_flow': 'load_inputs.car.CRITER_sup_35000dailyflow_3_4_5_lanes_flow',
    'CRITER_sup_35000dailyflow_3_4_5_lanes_occupancy': 'load_inputs.car.CRITER_sup_35000dailyflow_3_4_5_lanes_occupancy',
    'CRITER_inf_35000dailyflow_flow': 'load_inputs.car.CRITER_inf_35000dailyflow_flow',
    'CRITER_inf_35000dailyflow_occupancy': 'load_inputs.car.CRITER_inf_35000dailyflow_occupancy',
    'CRITER_urban_between_15000_25000dailyflow_flow': 'load_inputs.car.CRITER_urban_between_15000_25000dailyflow_flow',
    'CRITER_urban_between_15000_25000dailyflow_occupancy': 'load_inputs.car.CRITER_urban_between_15000_25000dailyflow_occupancy',
    'CRITER': 'load_inputs.car.CRITER',

    # NetMob: 
    'netmob': 'load_inputs.Lyon.netmob',
    'netmob_POIs': 'load_inputs.Lyon.netmob.netmob_POIs', 
    'netmob_POIs_per_station': 'load_inputs.Lyon.netmob_POIs_per_station',
    'netmob_image_per_station': 'load_inputs.Lyon.netmob_image_per_station',
    'netmob_video_lyon': 'load_inputs.Lyon.netmob_video_lyon',

    # Weather:
    'weather': 'load_inputs.Lyon.weather',
    # ---


    # --- Manhattan : 
    'METR_LA': 'load_inputs.Manhattan.METR_LA',
    'PEMS_BAY': 'load_inputs.Manhattan.PEMS_BAY',
    'PeMS03': 'load_inputs.Manhattan.PeMS03',
    'PeMS04': 'load_inputs.Manhattan.PeMS04',
    'PeMS07': 'load_inputs.Manhattan.PeMS07',
    'PeMS08': 'load_inputs.Manhattan.PeMS08',
    'PeMS08_flow': 'load_inputs.Manhattan.PeMS08_flow',
    'PeMS08_occupancy': 'load_inputs.Manhattan.PeMS08_occupancy',
    'PeMS08_speed': 'load_inputs.Manhattan.PeMS08_speed',
    
    }

''' NetMob Parameters: '''
#SELECTED_APPS = ['Google_Maps','Deezer','Instagram'] #,'Deezer','WhatsApp','Twitter'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
#TRANSFER_MODE = ['DL'] #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
#SELECTED_TAGS = ['station','iris','stadium','university']#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
#EXPANDED =  '' # '' # '_expanded'
 
#
# trafic_apps = ['Uber', 'Google_Maps','Waze'],
# music_apps = ['Spotify','Deezer','Apple_Music','Apple_iTunes','SoundCloud'],
# direct_messenger_apps = ['Telegram','Apple_iMessage','Facebook_Messenger','Snapchat','WhatsApp'],
# social_networks_apps = ['Twitter', 'Pinterest','Facebook','Instagram'],
#

# Load CRITER data : 
#FILE_NAME = 'preprocessed_CRITER_6min.csv'