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
import glob
import pickle
from build_inputs.load_netmob_data import find_positions,replace_heure_d_ete
''' This file has to :
 - return a DataSet object, with specified data, and spatial_units.
 - >>>> No Need to set n_vertex as it's a contextual data 
 - Detail 'INVALID_DATE' and the 'coverage' period of the dataset.
'''

FILE_NAME = 'netmob_image_per_station'
START = '03/16/2019'
END = '06/01/2019'
FREQ = '15min'

list_of_invalid_period = []
list_of_invalid_period.append([datetime(2019,5,16,0,0),datetime(2019,5,16,18,15)])  # 16 mai 00:00 - 18:15
list_of_invalid_period.append([datetime(2019,5,11,23,15),datetime(2019,5,12,0,0)])  # 11 mai 23:15 - 11 mai 23:59: down META (fb, whatsapp)
list_of_invalid_period.append([datetime(2019,5,23,0,0),datetime(2019,5,25,6,0)])  # Anoamlies for every single apps  23-25 May


## C = 1
## n_vertex = 

def load_data(dataset,ROOT,FOLDER_PATH,invalid_dates,intersect_coverage_period,args,normalize= True): # args,ROOT,FOLDER_PATH,coverage_period = None
    '''
    args:
    ------
    restricted : if True, then the used map is at least 4 times smaller and not sparse anymore
    normalize : supposed to be = True
    '''

    if torch.cuda.is_available():
        netmob_T = load_netmob_per_subway_station(ROOT,FOLDER_PATH,args,intersect_coverage_period,dataset.spatial_unit)

    else:
        raise NotImplementedError

    NetMob_ds = load_input_and_preprocess(dims = [0,3,4],normalize=normalize,invalid_dates=invalid_dates,args=args,netmob_T=netmob_T,dataset=dataset)
    return(NetMob_ds)


def load_netmob_per_subway_station(ROOT,FOLDER_PATH,args,intersect_coverage_period,columns,
                     trafic_apps = ['Uber', 'Google_Maps','Waze'],
                     music_apps = ['Spotify','Deezer','Apple_Music','Apple_iTunes','SoundCloud'],
                     direct_messenger_apps = ['Telegram','Apple_iMessage','Facebook_Messenger','Snapchat','WhatsApp'],
                     social_networks_apps = ['Twitter', 'Pinterest','Facebook','Instagram'],
                     ):
    '''Load NetMob Data:
    outputs:
    --------
    # NetMob Tensor : [T,N,C,H,W]
    # dims : [0,3,4] #[0,-2,-1]  -> dimension for which we want to retrieve stats 
    '''
    data_path = f"{ROOT}/{FOLDER_PATH}/{FILE_NAME}"
    # selected_apps =  ['Google_Maps'] #trafic_apps # music_apps  # direct_messenger_apps # social_networks_apps
    apps=  pickle.load(open(f'{data_path}/apps.pkl','rb'))#glob.glob(f'{ROOT}/{FOLDER_PATH}/{FILE_NAME}/*.pt')  

    # Extract only some specific apps: 
    trafic_pos = find_positions(args.NetMob_selected_apps,apps)

    # Extract only some specific date : 
    coverage_local = pd.date_range(start=START, end=END, freq=args.freq)[:-1]
    indices_dates = [k for k,date in enumerate(coverage_local) if date in intersect_coverage_period]

    if args.netmob_transfer_mode is None: 
        trafic_pos = [2*k for k in trafic_pos] + [2*k+1 for k in trafic_pos]
        print('Transfer Modes: DL and UL')
    elif args.netmob_transfer_mode == 'DL':
        trafic_pos = [2*k for k in trafic_pos] 
        print('Transfer Modes: DL')
    elif args.netmob_transfer_mode == 'UL':
        trafic_pos = [2*k+1 for k in trafic_pos]
        print('Transfer Modes: UL')
        
        
    assert len(apps) == 136//2 # Tensor.size(1) =nb_mode_transfer x nb_apps =2*68  = 136

    # Select specific apps 
    netmob_T = torch.stack([torch.load(f"{data_path}/station_{station}.pt")[:,trafic_pos,:,:] for station in columns])
    netmob_T = netmob_T.permute(1,0,*range(2, netmob_T.dim()))

    # Replace problematic time-slots:
    netmob_T = replace_heure_d_ete(netmob_T,start = 572, end = 576)

    if args.freq != FREQ :
        assert int(args.freq.replace('min',''))> int(FREQ.replace('min','')), f'Trying to apply a a {args.freq} temporal aggregation while the minimal possible one is {FREQ}'
        netmob_T = netmob_T.view(-1, int(args.freq.replace('min','')) // int(FREQ.replace('min','')), *netmob_T.shape[1:]).sum(dim=1)

    # Select specific dates : 
    netmob_T = netmob_T[indices_dates]

    return(netmob_T)


def load_input_and_preprocess(dims,normalize,invalid_dates,args,netmob_T,dataset):

    print('\nInit NetMob Dataset: ', netmob_T.size())
    print('Number of Nan Value: ',torch.isnan(netmob_T).sum())
    print('Total Number of Elements: ', netmob_T.numel(),'\n')

    NetMob_ds = PersonnalInput(invalid_dates,args, tensor = netmob_T, dates = dataset.df_dates,
                           time_step_per_hour = dataset.time_step_per_hour,
                           Weeks = args.W,
                           Days = args.D, 
                           historical_len = args.H,
                           step_ahead = args.step_ahead,
                            minmaxnorm = args.minmaxnorm,
                            standardize = args.standardize,
                           dims =dims,
                           data_augmentation= args.data_augmentation)
    NetMob_ds.preprocess(args.train_prop,args.valid_prop,args.test_prop,args.train_valid_test_split_method,normalize)

    return NetMob_ds