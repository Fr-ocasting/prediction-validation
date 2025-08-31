import sys 
import os 
import pandas as pd
import torch 

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
from datetime import datetime 
from pipeline.dataset import DataSet
from pipeline.dataset import PersonnalInput
import pickle 
from pipeline.build_inputs.load_contextual_data import find_positions,replace_heure_d_ete
from pipeline.build_inputs.load_preprocessed_dataset import load_input_and_preprocess
from constants.paths import SELECTED_APPS
''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - >>>> No Need to set num_nodes as it's a contextual data 
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''
NAME= 'netmob_video_lyon'
FILE_NAME = 'netmob_video_lyon'
START = '03/16/2019'
END = '06/01/2019'
USELESS_DATES = {'hour':[], #[1,2,3,4,5,6],  #[] if no useless (i.e removed) hours
                 'weekday':[]#[5,6],
                 }
FREQ = '15min'

list_of_invalid_period = []
list_of_invalid_period.append([datetime(2019,5,16,0,0),datetime(2019,5,16,18,15)])  # 16 mai 00:00 - 18:15
list_of_invalid_period.append([datetime(2019,5,11,23,15),datetime(2019,5,12,0,0)])  # 11 mai 23:15 - 11 mai 23:59: down META (fb, whatsapp)
list_of_invalid_period.append([datetime(2019,5,23,0,0),datetime(2019,5,25,6,0)])  # Anoamlies for every single apps  23-25 May

## C = 1
## num_nodes = 


def load_data(dataset,FOLDER_PATH,invalid_dates,args,minmaxnorm,standardize,restricted,normalize= True): # args,FOLDER_PATH,coverage_period = None
    '''
    args:
    ------
    restricted : if True, then the used map is at least 4 times smaller and not sparse anymore
    normalize : supposed to be = True
    '''

    if torch.cuda.is_available():
        netmob_T = load_netmob_lyon_map(FOLDER_PATH,restricted)
        if args.freq != FREQ :
            assert int(args.freq.replace('min',''))> int(FREQ.replace('min','')), f'Trying to apply a a {args.freq} temporal aggregation while the minimal possible one is {FREQ}'
            netmob_T = netmob_T.view(-1, int(args.freq.replace('min','')) // int(FREQ.replace('min','')), *netmob_T.shape[1:]).sum(dim=1)

    else:
        raise NotImplementedError

    NetMob_ds = load_input_and_preprocess(dims = [0,2,3],normalize=normalize,invalid_dates=invalid_dates,args=args,netmob_T=netmob_T,dataset=dataset,name=NAME,minmaxnorm=minmaxnorm,standardize=standardize)
    return(NetMob_ds)

   
def load_netmob_lyon_map(FOLDER_PATH,
                     trafic_apps = ['Uber', 'Google_Maps','Waze'],
                     music_apps = ['Spotify','Deezer','Apple_Music','Apple_iTunes','SoundCloud'],
                     direct_messenger_apps = ['Telegram','Apple_iMessage','Facebook_Messenger','Snapchat','WhatsApp'],
                     social_networks_apps = ['Twitter', 'Pinterest','Facebook','Instagram'],
                    restricted = True 
                     ):
    '''Load NetMob Data:
    outputs:
    --------
    # NetMob Tensor : [T,C,H,W]
    # dims : [0,3,4] #[0,-2,-1]  -> dimension for which we want to retrieve stats 
    '''

    #selected_apps = ['Uber','Google_Maps','Spotify','Instagram','Deezer','WhatsApp','Twitter','Snapchat']

    apps =  pickle.load(open(f"{FOLDER_PATH}/{FILE_NAME}/{FILE_NAME}_APP.pkl","rb"))
    trafic_pos = find_positions(SELECTED_APPS,apps)
    if restricted:
        netmob_T = torch.load(f"{FOLDER_PATH}/{FILE_NAME}/{FILE_NAME}.pt")[trafic_pos,:,110:-40,85:-55]
    else:
        netmob_T = torch.load(f"{FOLDER_PATH}/{FILE_NAME}/{FILE_NAME}.pt")[trafic_pos,:,:,:]                
    netmob_T = netmob_T.permute(1,0,2,3)

    # Replace problematic time-slots:
    netmob_T = replace_heure_d_ete(netmob_T,start = 572, end = 576)

    return(netmob_T)