import os 
import torch 

# Usual paths: 
if torch.cuda.is_available():
    FOLDER_PATH = '../../../../data' 
    # Load subway in data:
    #FILE_NAME = 'preprocessed_subway_15_min' #.csv
    #FILE_NAME = 'subway_IN_interpol_neg_15_min_16Mar2019_1Jun2020' #.csv
    #FILE_NAME = 'subway_IN_interpol_neg_15_min_2019_2020' #.csv
else:
    #current_path = os.path.abspath(os.path.dirname(__file__))
    #package_path = os.path.abspath(os.path.join(current_path, '..'))
    #FOLDER_PATH = '../../../../Data/'
    FOLDER_PATH =  '../data'
    DATA_TO_PREDICT = 'subway_in' # 'data_bidon'  # 'subway_in' 
    #FILE_NAME = 'data_bidon' #.csv

SAVE_DIRECTORY = '../save/'
ABS_PATH_PACKAGE = '/Users/rrochas/Desktop/Code/prediction-validation'

# Load CRITER data : 
#FILE_NAME = 'preprocessed_CRITER_6min.csv'

results_folder = f"{SAVE_DIRECTORY}results/"
if not(os.path.exists(results_folder)):
    os.makedirs(results_folder)