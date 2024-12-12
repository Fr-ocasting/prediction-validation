import torch
import glob 
import argparse
from argparse import Namespace
import pickle
# Relative path:
import sys 
import os 
import importlib
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
    elif tensor.dim() == 2:
        tensor[start:end,:] = mean_values.repeat(4,1)
    else:
        raise NotImplementedError(f'dim {tensor.dim()} has not been implemented')
    return tensor


def load_input_and_preprocess(dims,normalize,invalid_dates,args,netmob_T,dataset):

    print('\nInit NetMob Dataset: ', netmob_T.size())
    print('Number of Nan Value: ',torch.isnan(netmob_T).sum())
    print('Total Number of Elements: ', netmob_T.numel(),'\n')

    NetMob_ds = PersonnalInput(invalid_dates,args, tensor = netmob_T, dates = dataset.df_dates,
                           time_step_per_hour = dataset.time_step_per_hour,Weeks = args.W, Days = args.D, historical_len = args.H,step_ahead = args.step_ahead,minmaxnorm = True,dims =dims)
    NetMob_ds.preprocess(args.train_prop,args.valid_prop,args.test_prop,args.train_valid_test_split_method,normalize)

    return NetMob_ds

def tackle_input_data(dataset,invalid_dates,intesect_coverage_period,args,normalize):
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
        netmob_dataset_name = 'netmob_video_lyon'


    elif 'netmob_image_per_station' in args.dataset_names:
        from load_inputs.netmob_image_per_station import load_data
        NetMob_ds = load_data(dataset,parent_dir,FOLDER_PATH,invalid_dates,intesect_coverage_period,args,normalize = normalize) 
        args.vision_input_type = 'image_per_stations'
        netmob_dataset_name = 'netmob_image_per_station'
        
    elif "netmob_POIs" in args.dataset_names:
        from load_inputs.netmob_POIs import load_data
        NetMob_ds = load_data(dataset,parent_dir,FOLDER_PATH,invalid_dates,intesect_coverage_period,args,normalize= normalize)
        args.vision_input_type = 'POIs'
        netmob_dataset_name = 'netmob_POIs'

    else :
        raise NotImplementedError(f'load data has not been implemented for the netmob file here {args.dataset_names}')
    
    return(NetMob_ds,args,netmob_dataset_name)

def tackle_config_of_feature_extractor_module(NetMob_ds,args_vision):
    if args_vision.dataset_name == 'netmob_POIs':
        C_netmob = NetMob_ds[0].U_train.size(1) # [B,R]
        List_input_sizes = [NetMob_ds[k].U_train.size(2) for k in range(len(NetMob_ds)) ]
        List_nb_channels = [NetMob_ds[k].U_train.size(1) for k in range(len(NetMob_ds)) ]
        script = importlib.import_module(f"dl_models.vision_models.{args_vision.model_name}.load_config")
        importlib.reload(script) 
        config_vision =script.get_config(List_input_sizes,List_nb_channels)# script.get_config(C_netmob)
        args_vision = Namespace(**{**vars(config_vision),**vars(args_vision)})
    else: 
        C_netmob = NetMob_ds.U_train.size(2) if len(NetMob_ds.U_train.size())==6 else  NetMob_ds.U_train.size(1)# [B,N,C,H,W,L]  or [B,C,H,W,L] 
        H,W,L = NetMob_ds.U_train.size(-3),NetMob_ds.U_train.size(-2),NetMob_ds.U_train.size(-1)
        script = importlib.import_module(f"dl_models.vision_models.{args_vision.model_name}.load_config")
        importlib.reload(script) 
        config_vision = script.get_config(H,W,L)
        args_vision = Namespace(**{**vars(config_vision),**vars(args_vision)})


    # ===== Define Namespace 'args_vision': 
    # =====================================
    # FeatureExtractor_ResNetInspired
    if False : 
        if args_vision['model_name'] == 'FeatureExtractor_ResNetInspired':
            args_vision.update({'c_in' : C_netmob, 'h_dim': 128, 'L':L}) # out_dim = L*h_dim//2
        
        elif args_vision['model_name'] == 'FeatureExtractor_ResNetInspired_bis':
            args_vision.update({'c_in' : C_netmob, 'out_dim': 64}) 

        # MinimalFeatureExtractor  
        elif args_vision['model_name'] == 'MinimalFeatureExtractor':
            h_dim = 16
            args_vision.update({'c_in' : C_netmob,'h_dim': h_dim, 'L' : L}) # out_dim = L*h_dim//2

        
        # ImageAvgPooling
        #elif args_vision['model_name'] == 'ImageAvgPooling':
        #    args_vision.update({'out_dim' : L}) # out_dim = L
        #

        elif args_vision['model_name'] == 'FeatureExtractorEncoderDecoder':  # (c_in=3, z_dim=64, N=40)
            args_vision.update({'c_in' : C_netmob, 'out_dim': 64, 'H':H,'W':W,L:'L'}) 

        elif args_vision['model_name'] == 'AttentionFeatureExtractor': # (c_in=3, z_dim=64, N=40)
            args_vision.update({'c_in' : C_netmob, 'out_dim': 64}) 

        elif args_vision['model_name'] == 'VideoFeatureExtractorWithSpatialTemporalAttention': # (c_in=3, out_dim=64, N=40, d_model=128))
            args_vision.update({'c_in' : C_netmob, 'out_dim': 64, 'd_model':128}) 

        else:
            raise NotImplementedError(f"Model vision {args_vision['model_name']} has not been implemented")
        
    return args_vision


def tackle_netmob(dataset,invalid_dates,intesect_coverage_period,args,normalize = True):

    # BOOLEAN VALUE : True IF NETMOB IS USED
    bool_netmob = (sum([True for d in args.dataset_names if 'netmob' in d])) > 0
    
    if bool_netmob: 
        # TACKLE THE INPUT DATA 
        NetMob_ds,args,netmob_dataset_name = tackle_input_data(dataset,invalid_dates,intesect_coverage_period,args,normalize)

        # TACKLE THE FEATURE EXTRACTOR MODULE 
        print('vision_input_type', args.vision_input_type)
        print('vision_model_name', args.vision_model_name)
        if args.vision_model_name is None: raise ValueError("You are using 'NetMob' data but you didnot defined 'args.vision_model_name'. It needs to be set ")
        args_vision = Namespace(**{'dataset_name': netmob_dataset_name, 'model_name':args.vision_model_name,'input_type':args.vision_input_type})
        args_vision = tackle_config_of_feature_extractor_module(NetMob_ds,args_vision)
        args.args_vision = args_vision

        # Get args_vision:
        #parser = argparse.ArgumentParser(description='netmob')
        #for key,value in args_vision.items():
        #    parser.add_argument(f'--{key}', type=type(value), default=value)
        #args_vision = parser.parse_args(args=[])
        #args.args_vision = args_vision
        # ...
    else:
        NetMob_ds = None
        args.args_vision = argparse.ArgumentParser(description='netmob').parse_args(args=[])
    return args,NetMob_ds