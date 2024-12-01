import sys 
import os 
import pandas as pd
import torch 

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
from datetime import datetime 
from dataset import DataSet
from dataset import PersonnalInput
import numpy as np 
import pickle
from build_inputs.load_netmob_data import find_positions,replace_heure_d_ete
from constants.paths import SELECTED_APPS
''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - >>>> No Need to set n_vertex as it's a contextual data 
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''

EXPANDED =  '_expanded' # '' # '_expanded'
FILE_NAME = 'netmob_image_per_station'
list_of_invalid_period = []
list_of_invalid_period.append([datetime(2019,5,16,0,0),datetime(2019,5,16,18,15)])  # 16 mai 00:00 - 18:15
list_of_invalid_period.append([datetime(2019,5,11,23,15),datetime(2019,5,12,0,0)])  # 11 mai 23:15 - 11 mai 23:59: down META (fb, whatsapp)
list_of_invalid_period.append([datetime(2019,5,23,0,0),datetime(2019,5,25,6,0)])  # Anoamlies for every single apps  23-25 May


INVALID_DATES = []
for start,end in list_of_invalid_period:
    INVALID_DATES = INVALID_DATES + list(pd.date_range(start,end,freq = f'15min'))

## C = 1
## n_vertex = 
COVERAGE = pd.date_range(start='03/16/2019', end='06/1/2019', freq='15min')[:-1]

def load_data(dataset,ROOT,FOLDER_PATH,invalid_dates,intesect_coverage_period,args,normalize= True): # args,ROOT,FOLDER_PATH,coverage_period = None
    '''
    args:
    ------
    restricted : if True, then the used map is at least 4 times smaller and not sparse anymore
    normalize : supposed to be = True

    outputs:
    --------
    4-th order tensor [len(apps),len(osmid),len(transfer_modes),len(time-serie)]
    '''

    id_stations = dataset.spatial_unit
    NetMob_ds = []
    for id_station in id_stations:
        # data_app.shape :[len(apps),len(osmid_associated),len(transfer_modes),T]
        data_app = load_data_npy(id_station,ROOT,FOLDER_PATH,args)
        netmob_T = torch.Tensor(data_app)
        # netmob_T.shape : [T,len(apps),len(osmid_associated),len(transfer_modes)]
        netmob_T = netmob_T.permute(3,0,1,2)
        # netmob_T.shape : [T,len(apps)*len(osmid_associated)*len(transfer_modes)] = [T,R]
        netmob_T = netmob_T.reshape(netmob_T.size(0),-1)
        
        # Extract only usefull data, and replace "heure d'été"
        indices_dates = [k for k,date in enumerate(COVERAGE) if date in intesect_coverage_period]
        netmob_T = replace_heure_d_ete(netmob_T,start = 572, end = 576)
        # [T,R] -> [T,R']
        netmob_T = netmob_T[indices_dates]


        # dimension on which we want to normalize: 
        dims = [0]# [0]  -> We are normalizing each time-serie independantly 
        NetMob_POI = load_input_and_preprocess(dims = dims,normalize=normalize,invalid_dates=invalid_dates,args=args,netmob_T=netmob_T,dataset=dataset)
        NetMob_POI.station_name = id_station
        NetMob_ds.append(NetMob_POI)
    return(NetMob_ds)



def load_data_npy(id_station,ROOT,FOLDER_PATH,args):
    save_folder = f"{ROOT}/{FOLDER_PATH}/POIs/netmob_POI_Lyon{EXPANDED}/Inputs/{id_station}"
    data_app = np.load(open(f"{save_folder}/data.npy","rb"))
    metadata = pickle.load(open(f"{save_folder}/metadata.pkl","rb"))

    transfer_modes = metadata['transfer_modes']
    # If we select a specific transfer_modes: 
    if args.netmob_transfer_mode == ['DL']:
        if 'DL' == transfer_modes[0]:
            data_app = data_app[:,:,:1,:]  
        else:
            data_app = data_app[:,:,1:,:] 
    if args.netmob_transfer_mode == ['UL']:
        if 'UL' == transfer_modes[1]:
            data_app = data_app[:,:,1:,:]
        else :
            data_app = data_app[:,:,:1,:]       

    # If we select specific apps:
    apps = metadata['apps']
    data_app = reorganise_npy(data_app,apps,SELECTED_APPS)
    return(data_app)

def reorganise_npy(data_app,apps,SELECTED_APPS):
    ''' Allow to select the npy dataset with the corresponding to selected apps.'''
    def find_pos(app_i,apps):
        for k,app in enumerate(apps):
            if app == app_i:
                return k 
    list_pos = [find_pos(app_i,apps) for app_i in SELECTED_APPS]    
    return data_app[list_pos,:,:,:]



def load_input_and_preprocess(dims,normalize,invalid_dates,args,netmob_T,dataset):
    NetMob_ds = PersonnalInput(invalid_dates,args, tensor = netmob_T, dates = dataset.df_dates,
                           time_step_per_hour = dataset.time_step_per_hour,Weeks = args.W, Days = args.D, historical_len = args.H,step_ahead = args.step_ahead,minmaxnorm = True,dims =dims)
    NetMob_ds.preprocess(args.train_prop,args.valid_prop,args.test_prop,args.train_valid_test_split_method,normalize)

    return NetMob_ds