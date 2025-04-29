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
from build_inputs.load_datasets_to_predict import load_datasets_to_predict
from utils.utilities import get_time_step_per_hour

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


def load_input_and_preprocess(dims,normalize,invalid_dates,args,contextual_T,dataset=None,df_dates=None):

    print('\nInit contextual_ds Dataset: ', contextual_T.size())
    print('Number of Nan Value: ',torch.isnan(contextual_T).sum())
    print('Total Number of Elements: ', contextual_T.numel(),'\n')

    if df_dates is None:
        df_dates = dataset.df_dates

    contrextual_ds = PersonnalInput(invalid_dates,args, tensor = contextual_T, dates = df_dates,
                           time_step_per_hour = get_time_step_per_hour(args.time_step_per_hour),
                           Weeks = args.W, 
                           Days = args.D, 
                           historical_len = args.H,
                           step_ahead = args.step_ahead,
                           minmaxnorm = args.minmaxnorm,
                           standardize = args.standardize,
                           dims =dims,
                           data_augmentation= args.data_augmentation
                           )
    contrextual_ds.preprocess(args.train_prop,args.valid_prop,args.test_prop,args.train_valid_test_split_method,normalize)

    return contrextual_ds

def tackle_input_data(invalid_dates,coverage_period,args,normalize):
    ''' Load the contextual data

    args : 
    -----
    args.dataset_names
    >>>> if == 'netmob_video_lyon' : return a 4-th order torch tensor [T,C,H,W] with a unique image (~260x280) for the entire city Lyon
    >>>> if == 'netmob_image_per_station' : return a 5-th order torch tensor [T,C,N,H,W]  with an image around each subway station 
    
    '''
    contextual_ds = {}
    for dataset_name in args.contextual_dataset_names:
        print('\n>>>Tackle Contextual dataset: ',dataset_name)
        module_path = f"load_inputs.{dataset_name}"
        module = importlib.import_module(module_path)
        importlib.reload(module)
        contextual_ds_i = module.load_data(FOLDER_PATH,
                                           coverage_period = coverage_period,
                                           invalid_dates=invalid_dates,
                                           args=args,
                                           normalize=normalize
                                           )
        contextual_ds_i.name = dataset_name
        contextual_ds[dataset_name] = contextual_ds_i
        return(contextual_ds,args)
        

def tackle_config_of_feature_extractor_module(contextual_ds,args_vision):
    if len(vars(args_vision))>0:
        if (args_vision.dataset_name == 'netmob_POIs_per_station') or (args_vision.dataset_name == 'subway_out_per_station') :
            List_input_sizes = [contextual_ds[k].U_train.size(2) for k in range(len(contextual_ds)) ]
            List_nb_channels = [contextual_ds[k].U_train.size(1) for k in range(len(contextual_ds)) ]
            script = importlib.import_module(f"dl_models.vision_models.{args_vision.model_name}.load_config")
            importlib.reload(script) 
            config_vision =script.get_config(List_input_sizes,List_nb_channels)# script.get_config(C_netmob)
            args_vision = Namespace(**{**vars(config_vision),**vars(args_vision)})
        elif (args_vision.dataset_name == 'netmob_POIs')or (args_vision.dataset_name == 'subway_out') :
            input_size = contextual_ds.U_train.size(2)
            nb_channels = contextual_ds.U_train.size(1)
            script = importlib.import_module(f"dl_models.vision_models.{args_vision.model_name}.load_config")
            importlib.reload(script) 
            config_vision =script.get_config(input_size,nb_channels)# script.get_config(C_netmob)
            args_vision = Namespace(**{**vars(config_vision),**vars(args_vision)})

        else: 
            raise NotImplementedError(f"args_vision.dataset_name '{args_vision.dataset_name}' n'est probabelment pas importé correctement. Modifier le code.")
            C_netmob = NetMob_ds.U_train.size(2) if len(NetMob_ds.U_train.size())==6 else  NetMob_ds.U_train.size(1)# [B,N,C,H,W,L]  or [B,C,H,W,L] 
            H,W,L = NetMob_ds.U_train.size(-3),NetMob_ds.U_train.size(-2),NetMob_ds.U_train.size(-1)
            script = importlib.import_module(f"dl_models.vision_models.{args_vision.model_name}.load_config")
            importlib.reload(script) 
            config_vision = script.get_config(H,W,L)
            args_vision = Namespace(**{**vars(config_vision),**vars(args_vision)})

    return args_vision


def tackle_contextual(target_ds,invalid_dates,coverage_period,args,normalize = True):

    # Define contextual tensors
    contextual_dataset_names = [ds_name for ds_name in args.dataset_names if not (ds_name in (['calendar','calendar_embedding']+ [target_ds.target_data]))]
    if args.use_target_as_context:
        if target_ds.target_data not in contextual_dataset_names:
            contextual_dataset_names.append(target_ds.target_data)
    args.contextual_dataset_names = contextual_dataset_names

    # USE CONTEXTUAL DATA
    if len(args.contextual_dataset_names) > 0: 

        contextual_ds,args = tackle_input_data(invalid_dates,coverage_period,args,normalize)

        # TACKLE THE FEATURE EXTRACTOR MODULE 
        if args.vision_model_name is None: 
            if not args.stacked_contextual:
                raise ValueError("You are using contextual data but you did not defined 'args.vision_model_name'. It needs to be set ")
            else:
                args.args_vision = argparse.ArgumentParser(description='args_vision').parse_args(args=[])
                if (args.compute_node_attr_with_attn) or (sum(['per_station' in contextual_ds_i.name for name_i,contextual_ds_i in contextual_ds.items()])>0): 
                    scrip_args = importlib.import_module(f"dl_models.SpatialAttn.load_config")
                    importlib.reload(scrip_args)
                    latent_dim = scrip_args.args.latent_dim 
                    raise NotImplementedError(f"VA PRENDRE EN COMPTE UNE SPATIAL ATTENTION MAIS NE SAIS PAS POUR QUEL DONNEE LA FAIRE")
                else:
                    latent_dim = 1
                if type(contextual_ds)==list:
                    add_C = latent_dim*contextual_ds[0].C
                else:
                    if ('netmob_POIs' in args.dataset_names) and (args.stacked_contextual) and (not args.compute_node_attr_with_attn):
                        add_C = len(args.NetMob_selected_apps)*len(args.NetMob_transfer_mode)*len(args.NetMob_selected_tags) 
                        raise NotImplementedError("NE PREND PAS EN COMPTE CORRECTEMENT NETMOB POIS AVEC D AUTRES DONNEE CONTUEXTUELLES")
                    else:
                        add_C = latent_dim*sum([contextual_ds_i.C for name_i,contextual_ds_i in contextual_ds.items()])
                args.C = args.C + add_C

        else:
            if args.stacked_contextual:
                raise ValueError("You defined a feature extractor model from your contextual data but you plan to stack the contextual in a channel.\n\
                                  It's not consistent as with 'stacked_contextual' you are not supposed to extract feature information before the core model.\n\
                                 Otherwise, set 'stacked_contextual' to False\
                                 ")
            else:
                print('   vision_input_type', args.vision_input_type)
                print('   vision_model_name', args.vision_model_name)
                args_vision = Namespace(**{'dataset_name': [contextual_ds_i.name for  name_i,contextual_ds_i in contextual_ds.items()], 'model_name':args.vision_model_name,'input_type':args.vision_input_type})
                args_vision = tackle_config_of_feature_extractor_module(contextual_ds,args_vision)
                args.args_vision = args_vision
        print(f"   Init Dataset: '{[c_i.raw_values.size()for _,c_i in contextual_ds.items()]}. {[torch.isnan(c_i.raw_values).sum()for _,c_i in contextual_ds.items()]} Nan values")
        print('   TRAIN contextual_ds:',[c_i.U_train.size() for _,c_i in contextual_ds.items()])
        print('   VALID contextual_ds:',[c_i.U_valid.size() for  _,c_i in contextual_ds.items()]) if hasattr(target_ds,'U_valid') else None
        print('   TEST contextual_ds:',[c_i.U_test.size() for  _,c_i in contextual_ds.items()]) if hasattr(target_ds,'U_test') else None


    else:
        contextual_ds = None
        args.args_vision = argparse.ArgumentParser(description='args_vision').parse_args(args=[])



    return args,contextual_ds