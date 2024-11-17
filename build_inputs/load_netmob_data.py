import torch
import glob 
import argparse
import pickle
# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal inputs:
from dataset import PersonnalInput
from constants.paths import FOLDER_PATH


def find_positions(applications, file_list):
    positions = []
    for app in applications:
        for idx, file_path in enumerate(file_list):
            FILE_NAME = file_path.split('/')[-1].split('.')[0]
            if app == FILE_NAME:
                positions.append(idx)
    return positions

def replace_heure_d_ete(tensor,start = 572, end = 576):
    values_before = tensor[start-1:start]
    values_after = tensor[end:end+1]

    mean_values = (values_before + values_after) / 2

    if tensor.dim() == 5:
        mean_values = mean_values.repeat(4,1,1,1,1)
        tensor[start:end,:,:,:,:] = mean_values
    elif tensor.dim() == 4:
        mean_values = mean_values.repeat(4,1,1,1)
        tensor[start:end,:,:,:] = mean_values
    else:
        raise NotImplementedError(f'dim {tensor.dim} has not been implemented')
    return tensor
"""
def load_netmob_data(dataset,invalid_dates,args,columns,
                     trafic_apps = ['Uber', 'Google_Maps','Waze'],
                     music_apps = ['Spotify','Deezer','Apple_Music','Apple_iTunes','SoundCloud'],
                     direct_messenger_apps = ['Telegram','Apple_iMessage','Facebook_Messenger','Snapchat','WhatsApp'],
                     social_networks_apps = ['Twitter', 'Pinterest','Facebook','Instagram'],
                     normalize = True,
                     ):
    '''Load NetMob Data:
    outputs:
    --------
    # NetMob Tensor : [T,N,C,H,W]
    # dims : [0,3,4] #[0,-2,-1]  -> dimension for which we want to retrieve stats 
    '''

    selected_apps = ['Google_Maps'] #trafic_apps # music_apps  # direct_messenger_apps # social_networks_apps
    dims = [0,3,4] 
     
    if torch.cuda.is_available():
        if args.quick_vision :
            netmob_T = torch.randn(dataset.length,40,4,22,22)
            print('Small NetMob_T ',netmob_T.size())
        else:
            apps=  glob.glob(f'{FOLDER_PATH}NetMob_tensor/[!station]*.pt')
            print(apps[0])
            trafic_pos = find_positions(selected_apps,apps)
            print('Trafic pos: ',trafic_pos)
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
            netmob_T = torch.stack([torch.load(f"{FOLDER_PATH}NetMob_tensor/station_{station}.pt")[:,trafic_pos,:,:] for station in columns])
            netmob_T = netmob_T.permute(1,0,*range(2, netmob_T.dim()))

            # Replace problematic time-slots:
            netmob_T = replace_heure_d_ete(netmob_T,start = 572, end = 576)

        # Keep only time-slots associated to the dataset:
        if dataset.time_slot_limits is not None: netmob_T = netmob_T[dataset.time_slot_limits]

    else:
        netmob_T = torch.randn(dataset.length,40,2,8,8)  # (7400,40,67,22,22)
        print("Load des données NetMob .pt impossible. Création d'un random Tensor")

    NetMob_ds = load_input_and_preprocess(dims,normalize,invalid_dates,args,netmob_T,dataset)

    return(NetMob_ds)
"""

"""
def load_netmob_lyon_map(dataset,invalid_dates,args,columns,
                     trafic_apps = ['Uber', 'Google_Maps','Waze'],
                     music_apps = ['Spotify','Deezer','Apple_Music','Apple_iTunes','SoundCloud'],
                     direct_messenger_apps = ['Telegram','Apple_iMessage','Facebook_Messenger','Snapchat','WhatsApp'],
                     social_networks_apps = ['Twitter', 'Pinterest','Facebook','Instagram'],
                     normalize = True,
                         restricted = True 
                     ):
    '''Load NetMob Data:
    outputs:
    --------
    # NetMob Tensor : [T,C,H,W]
    # dims : [0,3,4] #[0,-2,-1]  -> dimension for which we want to retrieve stats 
    '''

    selected_apps = ['Uber','Google_Maps','Spotify','Instagram','Deezer','WhatsApp','Twitter','Snapchat']
    dims = [0,2,3]

    if torch.cuda.is_available():
        if args.quick_vision :
            netmob_T = torch.randn(dataset.length,4,80,80)
            print('Small NetMob_T ',netmob_T.size())
            
        else:
            apps =  pickle.load(open(f"{FOLDER_PATH}/NetMob_DL_video_Lyon_APP.pkl","rb"))
            trafic_pos = find_positions(selected_apps,apps)
            if restricted:
                netmob_T = torch.load(f"{FOLDER_PATH}/NetMob_DL_video_Lyon.pt")[trafic_pos,:,110:-40,85:-55]
            else:
                netmob_T = torch.load(f"{FOLDER_PATH}/NetMob_DL_video_Lyon.pt")[trafic_pos,:,:,:]                
            netmob_T = netmob_T.permute(1,0,2,3)

            # Replace problematic time-slots:
            netmob_T = replace_heure_d_ete(netmob_T,start = 572, end = 576)

        # Keep only time-slots associated to the dataset:
        if dataset.time_slot_limits is not None: netmob_T = netmob_T[dataset.time_slot_limits]

    else:
        netmob_T = torch.randn(dataset.length,4,22,22)  # (7392,4,22,22)
        print("Load des données NetMob .pt impossible. Création d'un random Tensor")

    NetMob_ds = load_input_and_preprocess(dims,normalize,invalid_dates,args,netmob_T,dataset)

    return(NetMob_ds)
"""
def load_input_and_preprocess(dims,normalize,invalid_dates,args,netmob_T,dataset):

    print('\nInit NetMob Dataset: ', netmob_T.size())
    print('Number of Nan Value: ',torch.isnan(netmob_T).sum())
    print('Total Number of Elements: ', netmob_T.numel(),'\n')

    NetMob_ds = PersonnalInput(invalid_dates,args, tensor = netmob_T, dates = dataset.df_dates,
                           time_step_per_hour = dataset.time_step_per_hour,Weeks = args.W, Days = args.D, historical_len = args.H,step_ahead = args.step_ahead,minmaxnorm = True,dims =dims)
    NetMob_ds.preprocess(args.train_prop,args.valid_prop,args.test_prop,args.train_valid_test_split_method,normalize)

    return NetMob_ds

def tackle_input_data(dataset,invalid_dates,intesect_coverage_period,args,columns,normalize):
    ''' Load the NetMob input data

    args : 
    -----
    args.dataset_names
    >>>> if == 'netmob_video_lyon' : return a 4-th order torch tensor [T,C,H,W] with a unique image (~260x280) for the entire city Lyon
    >>>> if == 'netmob_image_per_station' : return a 5-th order torch tensor [T,C,N,H,W]  with an image around each subway station 
    
    '''

    if 'netmob_video_lyon' in args.dataset_names:
    # if vision_input_type == 'unique_image_through_lyon':
        #NetMob_ds = load_netmob_lyon_map(dataset,invalid_dates,args,columns = columns,normalize = normalize)
        from load_inputs.netmob_video_lyon import load_data
        NetMob_ds = load_data(dataset,parent_dir,invalid_dates,intesect_coverage_period,args,restricted,normalize= True)
        args.vision_input_type = 'unique_image_through_lyon'


    elif 'netmob_image_per_station' in args.dataset_names:
        from load_inputs.netmob_image_per_station import load_data
        NetMob_ds = load_data(dataset,parent_dir,FOLDER_PATH,invalid_dates,intesect_coverage_period,args,columns = columns, normalize = normalize) 
        args.vision_input_type = 'image_per_stations'

    else :
        raise NotImplementedError(f'load data has not been implemented for the netmob file here {args.dataset_names}')
    
    return(NetMob_ds,args)

def tackle_config_of_feature_extractor_module(NetMob_ds,args_vision):

    C_netmob = NetMob_ds.U_train.size(2) if len(NetMob_ds.U_train.size())==6 else  NetMob_ds.U_train.size(1)# [B,N,C,H,W,L]  or [B,C,H,W,L] 
    H,W,L = NetMob_ds.U_train.size(-3),NetMob_ds.U_train.size(-2),NetMob_ds.U_train.size(-1)

    # Define Namespace 'args_vision': 
    

    # FeatureExtractor_ResNetInspired
    if args_vision['model_name'] == 'FeatureExtractor_ResNetInspired':
        args_vision.update({'c_in' : C_netmob, 'h_dim': 128, 'L':L}) # out_dim = L*h_dim//2
    
    elif args_vision['model_name'] == 'FeatureExtractor_ResNetInspired_bis':
        args_vision.update({'c_in' : C_netmob, 'out_dim': 64}) 

    # MinimalFeatureExtractor  
    elif args_vision['model_name'] == 'MinimalFeatureExtractor':
        h_dim = 16
        args_vision.update({'c_in' : C_netmob,'h_dim': h_dim, 'L' : L}) # out_dim = L*h_dim//2

    # ImageAvgPooling
    elif args_vision['model_name'] == 'ImageAvgPooling':
        args_vision.update({'out_dim' : L}) # out_dim = L

    elif args_vision['model_name'] == 'FeatureExtractorEncoderDecoder':  # (c_in=3, z_dim=64, N=40)
        args_vision.update({'c_in' : C_netmob, 'out_dim': 64, 'H':H,'W':W,L:'L'}) 

    elif args_vision['model_name'] == 'AttentionFeatureExtractor': # (c_in=3, z_dim=64, N=40)
        args_vision.update({'c_in' : C_netmob, 'out_dim': 64}) 

    elif args_vision['model_name'] == 'VideoFeatureExtractorWithSpatialTemporalAttention': # (c_in=3, out_dim=64, N=40, d_model=128))
        args_vision.update({'c_in' : C_netmob, 'out_dim': 64, 'd_model':128}) 

    else:
        raise NotImplementedError(f"Model vision {args_vision['model_name']} has not been implemented")
    
    return args_vision


def tackle_netmob(dataset,invalid_dates,intesect_coverage_period,args,columns,vision_model_name,normalize = True):
    if True: 
        # TACKLE THE INPUT DATA 
        NetMob_ds,args = tackle_input_data(dataset,invalid_dates,intesect_coverage_period,args,columns,normalize)

        # TACKLE THE FEATURE EXTRACTOR MODULE 
        print('vision_input_type', args.vision_input_type)
        print('vision_model_name', vision_model_name)
        args_vision = {'model_name':vision_model_name,'input_type':args.vision_input_type}
        args_vision = tackle_config_of_feature_extractor_module(NetMob_ds,args_vision)

        # Get args_vision:
        parser = argparse.ArgumentParser(description='netmob')
        for key,value in args_vision.items():
            parser.add_argument(f'--{key}', type=type(value), default=value)
        args_vision = parser.parse_args(args=[])
        args.args_vision = args_vision
        # ...
    else:
        NetMob_ds = None
        args.args_vision = argparse.ArgumentParser(description='netmob').parse_args(args=[])
    return args,NetMob_ds