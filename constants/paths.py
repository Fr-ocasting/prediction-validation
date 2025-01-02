import os 
import torch 

# Usual paths: 
if torch.cuda.is_available():
    FOLDER_PATH = '../../../../data/rrochas/prediction_validation' 
    # Load the target data to predict: 
    DATA_TO_PREDICT = 'subway_in' # 'data_bidon'  # 'subway_in' # 'METR_LA' # 'PEMS_BAY'
    ABS_PATH_PACKAGE = '/home/rrochas/prediction-validation'
    #FILE_NAME = 'preprocessed_subway_15_min' #.csv
    #FILE_NAME = 'subway_IN_interpol_neg_15_min_16Mar2019_1Jun2020' #.csv
    #FILE_NAME = 'subway_IN_interpol_neg_15_min_2019_2020' #.csv
else:
    #current_path = os.path.abspath(os.path.dirname(__file__))
    #package_path = os.path.abspath(os.path.join(current_path, '..'))
    #FOLDER_PATH = '../../../../Data/'
    FOLDER_PATH =  '../data'
    DATA_TO_PREDICT = 'subway_in' # 'data_bidon'  # 'subway_in' # 'METR_LA' # 'PEMS_BAY'
    ABS_PATH_PACKAGE = '/Users/rrochas/Desktop/Code/prediction-validation'
    #FILE_NAME = 'data_bidon' #.csv
    
SAVE_DIRECTORY = '../save/'

''' Training Parameters'''
USELESS_DATES = {'hour':[1,2,3,4,5,6],
                 'weekday':[5,6],
                 }

''' Calendar Parameters: '''
CALENDAR_TYPE=['dayofweek','hour']#['dayofweek', 'hour', 'minute', 'bank_holidays', 'school_holidays', 'remaining_holidays']

''' NetMob Parameters: '''
SELECTED_APPS = ['Instagram','Google_Maps'] #['Google_Maps']# ['Instagram','Google_Maps','Twitter']
TRANSFER_MODE = ['DL'] # ['DL'] # ['UL'] #['DL','UL']
SELECTED_TAGS = ['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit']#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
EXPANDED =  '' # '' # '_expanded'
EPSILON = 0.3  # Distance max for Agglomerative Cluster based on distance correlation 
#
# trafic_apps = ['Uber', 'Google_Maps','Waze'],
# music_apps = ['Spotify','Deezer','Apple_Music','Apple_iTunes','SoundCloud'],
# direct_messenger_apps = ['Telegram','Apple_iMessage','Facebook_Messenger','Snapchat','WhatsApp'],
# social_networks_apps = ['Twitter', 'Pinterest','Facebook','Instagram'],
#

# Load CRITER data : 
#FILE_NAME = 'preprocessed_CRITER_6min.csv'

results_folder = f"{SAVE_DIRECTORY}results/"
if not(os.path.exists(results_folder)):
    os.makedirs(results_folder)