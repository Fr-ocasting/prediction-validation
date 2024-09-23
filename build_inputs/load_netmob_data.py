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


def find_positions(applications, file_list):
    positions = []
    for app in applications:
        for idx, file_path in enumerate(file_list):
            if app in file_path:
                positions.append(idx)
    return positions

def replace_heure_d_ete(tensor,start = 572, end = 576):
    values_before = tensor[start-1:start]
    values_after = tensor[end:end+1]

    mean_values = (values_before + values_after) / 2
    mean_values = mean_values.repeat(4,1,1,1,1)

    tensor[start:end,:,:,:,:] = mean_values
    return tensor

def load_netmob_data(dataset,invalid_dates,args,folder_path,columns,
                     trafic_apps = ['Uber', 'Google_Maps','Waze'],
                     music_apps = ['Spotify','Deezer','Apple_Music','Apple_iTunes','SoundCloud'],
                     direct_messenger_apps = ['Telegram','Apple_iMessage','Facebook_Messenger','Snapchat','WhatsApp'],
                     social_networks_apps = ['Twitter', 'Pinterest','Facebook','Instagram'],
                     normalize = True
                     ):
    '''Load NetMob Data:
    outputs:
    --------
    # NetMob Tensor : [T,N,C,H,W]
    # dims : [0,3,4] #[0,-2,-1]  -> dimension for which we want to retrieve stats 
    '''

    selected_apps = trafic_apps # music_apps  # direct_messenger_apps # social_networks_apps
    dims = [0,3,4]
     
    if torch.cuda.is_available():
        apps=  glob.glob(f'{folder_path}NetMob_tensor/[!station]*.pt')
        trafic_pos = find_positions(selected_apps,apps)
        trafic_pos = [2*k for k in trafic_pos] + [2*k+1 for k in trafic_pos]
        assert len(apps) == 136//2 # Tensor.size(1) =nb_mode_transfer x nb_apps =2*68  = 136


        netmob_T = torch.stack([torch.load(f"{folder_path}NetMob_tensor/station_{station}.pt")[:,trafic_pos,:,:] for station in columns])
        '''
        if args.time_slot_limit is not None:
            netmob_T = torch.stack([torch.load(f"{folder_path}NetMob_tensor/station_{station}.pt")[:args.time_slot_limit,trafic_pos,:,:] for station in columns])
        else:
            netmob_T = torch.stack([torch.load(f"{folder_path}NetMob_tensor/station_{station}.pt")[:,trafic_pos,:,:] for station in columns])
        '''
            
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


def load_netmob_lyon_map(dataset,invalid_dates,args,folder_path,columns,
                     trafic_apps = ['Uber', 'Google_Maps','Waze'],
                     music_apps = ['Spotify','Deezer','Apple_Music','Apple_iTunes','SoundCloud'],
                     direct_messenger_apps = ['Telegram','Apple_iMessage','Facebook_Messenger','Snapchat','WhatsApp'],
                     social_networks_apps = ['Twitter', 'Pinterest','Facebook','Instagram'],
                     normalize = True
                     ):
    '''Load NetMob Data:
    outputs:
    --------
    # NetMob Tensor : [T,C,H,W]
    # dims : [0,3,4] #[0,-2,-1]  -> dimension for which we want to retrieve stats 
    '''

    selected_apps = ['Uber','Google_Maps','Spotify','Instagram']
    dims = [0,2,3]

     #T00[[0,1,6]].sum(0)
    if torch.cuda.is_available():
        name_save = 'NetMob_DL_video_Lyon'
        save_path = '../../../../data'
        netmob_T = torch.load(f'{folder_path}{name_save}.pt')  # [68, 7392, 263, 287]
        apps = pickle.load(open(f"{save_path}/{name_save}_APP.pkl",'rb'))
        trafic_pos = find_positions(selected_apps,apps)
        netmob_T = netmob_T[trafic_pos]    # [C, 7392, 263, 287]     -> here C = len(selected_apps) =  4

        # Re-organize Tensor
        netmob_T = netmob_T.permute(1,0,*range(2, netmob_T.dim()))  #[7392,C, 263, 287]

        # Replace problematic time-slots:
        netmob_T = replace_heure_d_ete(netmob_T,start = 572, end = 576)

        # Keep only time-slots associated to the dataset:
        if dataset.time_slot_limits is not None: netmob_T = netmob_T[dataset.time_slot_limits]

    else:
        netmob_T = torch.randn(dataset.length,4,2,22,22)  # (7392,4,22,22)
        print("Load des données NetMob .pt impossible. Création d'un random Tensor")

    NetMob_ds = load_input_and_preprocess(dims,normalize,invalid_dates,args,netmob_T,dataset)

    return(NetMob_ds)


def load_input_and_preprocess(dims,normalize,invalid_dates,args,netmob_T,dataset):

    print('\nInit NetMob Dataset: ', netmob_T.size())
    print('Number of Nan Value: ',torch.isnan(netmob_T).sum())
    print('Total Number of Elements: ', netmob_T.numel(),'\n')

    NetMob_ds = PersonnalInput(invalid_dates,args, tensor = netmob_T, dates = dataset.df_dates,
                           time_step_per_hour = dataset.time_step_per_hour,Weeks = args.W, Days = args.D, historical_len = args.H,step_ahead = args.step_ahead,minmaxnorm = True,dims =dims)
    NetMob_ds.preprocess(args.train_prop,args.valid_prop,args.test_prop,normalize)

    return NetMob_ds





def tackle_netmob(dataset,dataset_names,invalid_dates,args,folder_path,columns,vision_model_name,normalize = True):

    vision_input_type = args.vision_input_type

    if 'netmob' in dataset_names:
        if vision_input_type == 'image_per_stations':
            NetMob_ds = load_netmob_data(dataset,invalid_dates,args,folder_path,columns = columns, normalize = normalize)
        elif vision_input_type == 'unique_image_through_lyon':
            NetMob_ds = load_netmob_lyon_map(dataset,invalid_dates,args,folder_path,columns = columns,normalize = normalize)
        else:
            raise NotImplementedError(f'{vision_input_type} has not been implemented')

        C_netmob = NetMob_ds.U_train.size(2) if len(NetMob_ds.U_train.size())==5 else  NetMob_ds.U_train.size(1)# [B,N,C,H,W,L]  or [B,C,H,W,L] 
        L = NetMob_ds.U_train.size(-1)

        # Define Namespace 'args_vision': 
        args_vision = {'model_name':vision_model_name,'input_type':vision_input_type}

        # FeatureExtractor_ResNetInspired
        if vision_model_name == 'FeatureExtractor_ResNetInspired':
            args_vision.update({'c_in' : C_netmob, 'h_dim': 128, 'L':L})

        # MinimalFeatureExtractor  
        elif vision_model_name == 'MinimalFeatureExtractor':
            h_dim = 16
            args_vision.update({'c_in' : C_netmob,'h_dim': h_dim, 'L' : L})

        # ImageAvgPooling
        elif vision_model_name == 'ImageAvgPooling':
            args_vision.update({'out_dim' : L})

        else:
            raise NotImplementedError(f"Model vision {vision_model_name} has not been implemented")
        # ...

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