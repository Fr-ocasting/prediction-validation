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
from pipeline.DataSet.dataset import PersonnalInput
from constants.paths import FOLDER_PATH,DATASET_IMPORT_PATHS
from pipeline.utils.utilities import get_time_step_per_hour

def find_positions(applications, file_list):
    positions = []
    for app in applications:
        for idx, file_path in enumerate(file_list):
            FILE_NAME = file_path.split('/')[-1].split('.')[0]
            if app == FILE_NAME:
                positions.append(idx)
    return positions

def replace_heure_d_ete(tensor,start = 1532, end = 1536):
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

def get_common_dates_between_contextual_and_target(target_ds,contextual_ds,training_mode):
    L_df_verif =  [getattr(target_ds.tensor_limits_keeper,f'df_verif_{training_mode}')] +[getattr(contextual_ds_i.tensor_limits_keeper,f'df_verif_{training_mode}') for _,contextual_ds_i in contextual_ds.items()]
    common_dates = list(set.intersection(*map(set, [df_verif_i.iloc[:,-1] for df_verif_i in L_df_verif])))
    return common_dates

def restrain_to_common_dates(ds,training_mode,common_dates):
    #print(f"U{training_mode} before: {getattr(ds,f'U_{training_mode}').size()}")

    mask_df = getattr(ds.tensor_limits_keeper,f'df_verif_{training_mode}').iloc[:,-1].isin(common_dates)
    mask_tensor =  getattr(ds.tensor_limits_keeper,f'df_verif_{training_mode}').reset_index(drop=True).iloc[:,-1].isin(common_dates)
    mask_tensor = mask_tensor[mask_tensor].index 
    #print('mask_df:',mask_df.shape)
    #print(mask_df.head(5))
    #print('mask_tensor:',mask_tensor.shape)
    #print(mask_tensor[:5])
    setattr(ds.tensor_limits_keeper,f'df_verif_{training_mode}',getattr(ds.tensor_limits_keeper,f'df_verif_{training_mode}')[mask_df])
    setattr(ds,f'U_{training_mode}',getattr(ds,f'U_{training_mode}')[mask_tensor])
    setattr(ds,f'Utarget_{training_mode}',getattr(ds,f'Utarget_{training_mode}')[mask_tensor])

    #print(f"U{training_mode} after: {getattr(ds,f'U_{training_mode}').size()}")
    return ds 

def restrain_all_ds_to_common_dates(target_ds,contextual_ds):
    ''' Restrain all datasets to the common dates '''
    print('\n')
    for training_mode in ['train','valid','test']:
        if hasattr(target_ds,f'U_{training_mode}'):
            locals()[f'common_dates_{training_mode}'] = get_common_dates_between_contextual_and_target(target_ds,contextual_ds,training_mode)

            if locals()[f'common_dates_{training_mode}'] != len(getattr(target_ds.tensor_limits_keeper,f'df_verif_{training_mode}').iloc[:,-1].unique()):
                print(f"Restraining all datasets to {training_mode} common dates: {len(locals()[f'common_dates_{training_mode}'])} dates")
                
                # Apply to Target Data:
                target_ds = restrain_to_common_dates(target_ds,training_mode,locals()[f'common_dates_{training_mode}'])

                # Apply to Contextual Data
                for name_i,contextual_ds_i in contextual_ds.items():
                    contextual_ds[name_i] = restrain_to_common_dates(contextual_ds_i,training_mode,locals()[f'common_dates_{training_mode}'])

    return target_ds,contextual_ds
    
def tackle_input_data(target_ds,invalid_dates,coverage_period,args,normalize):
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
        module_path = f"{DATASET_IMPORT_PATHS[dataset_name]}"
        module = importlib.import_module(module_path)
        importlib.reload(module)

        ## ---
        # Init 
        standardize,minmaxnorm = args.standardize, args.minmaxnorm
        args_copy = Namespace(**vars(args))

        # If specific normalisation for a contextual ds:
        if hasattr(args,'contextual_kwargs') and (dataset_name in args.contextual_kwargs.keys()):
            if 'standardize' in args.contextual_kwargs[dataset_name].keys():
                standardize = args.contextual_kwargs[dataset_name]['standardize']
            else:
                standardize = False
            if 'minmaxnorm' in args.contextual_kwargs[dataset_name].keys():
                minmaxnorm = args.contextual_kwargs[dataset_name]['minmaxnorm']
            else:
                minmaxnorm = False
            # If no specific normalisation, go back to Init
            if not(minmaxnorm) and not(standardize):
                standardize,minmaxnorm = args.standardize, args.minmaxnorm
            if 'H' in args.contextual_kwargs[dataset_name].keys():
                args_copy.H = args.contextual_kwargs[dataset_name]['H']
            if 'D' in args.contextual_kwargs[dataset_name].keys():
                args_copy.D = args.contextual_kwargs[dataset_name]['D']
            if 'W' in args.contextual_kwargs[dataset_name].keys():
                args_copy.W = args.contextual_kwargs[dataset_name]['W']
            args_copy.contextual_kwargs[dataset_name]['loading_contextual_data'] = True
        ## ---

        
        # print('len coverage period: ',coverage_period)
        # print('len invalid_dates: ',invalid_dates)
        contextual_ds_i = module.load_data(FOLDER_PATH,
                                        coverage_period = coverage_period,
                                        invalid_dates=invalid_dates,
                                        args=args_copy,
                                        minmaxnorm = minmaxnorm,
                                        standardize = standardize,
                                        normalize=normalize,
                                        tensor_limits_keeper = target_ds.tensor_limits_keeper if hasattr(target_ds,'tensor_limits_keeper') else None,
                                        )
        if not hasattr(contextual_ds_i,'C'):
            contextual_ds_i.C = module.C


        contextual_ds[dataset_name] = contextual_ds_i

        ### Update args with contextual dataset information:
        if hasattr(args,'contextual_kwargs') and (dataset_name in args.contextual_kwargs.keys()):
            if hasattr(contextual_ds_i,'list_correspondence') : args.contextual_kwargs[dataset_name]['list_correspondence'] = contextual_ds_i.list_correspondence
            if hasattr(contextual_ds_i,'dictionnary_aggregated_iris') : args.contextual_kwargs[dataset_name]['dictionnary_aggregated_iris'] = contextual_ds_i.dictionnary_aggregated_iris
            if hasattr(contextual_ds_i,'dict_label2agg') : args.contextual_kwargs[dataset_name]['dict_label2agg'] = contextual_ds_i.dict_label2agg
            if hasattr(contextual_ds_i,'kept_zones') : args.contextual_kwargs[dataset_name]['kept_zones'] = contextual_ds_i.kept_zones
            args.contextual_kwargs[dataset_name]['C'] = contextual_ds_i.C
            args.contextual_kwargs[dataset_name]['spatial_unit'] = contextual_ds_i.spatial_unit
            args.contextual_kwargs[dataset_name]['n_spatial_unit'] = len(contextual_ds_i.spatial_unit) 
    ### Match the dates of the contextual datasets with the target dataset if differents: 
    target_ds,contextual_ds = restrain_all_ds_to_common_dates(target_ds,contextual_ds)

    return(contextual_ds,args)
        

def tackle_config_of_feature_extractor_module(contextual_ds,args_vision):
    if len(vars(args_vision))>0:
        if (args_vision.dataset_name == 'netmob_POIs_per_station') or (args_vision.dataset_name == 'subway_out_per_station') :
            List_input_sizes = [contextual_ds[k].U_train.size(2) for k in range(len(contextual_ds)) ]
            List_nb_channels = [contextual_ds[k].U_train.size(1) for k in range(len(contextual_ds)) ]
            script = importlib.import_module(f"pipeline.dl_models.vision_models.{args_vision.model_name}.load_config")
            importlib.reload(script) 
            config_vision =script.get_config(List_input_sizes,List_nb_channels)# script.get_config(C_netmob)
            args_vision = Namespace(**{**vars(config_vision),**vars(args_vision)})
        elif (args_vision.dataset_name == 'netmob_POIs')or (args_vision.dataset_name == 'subway_out') :
            input_size = contextual_ds.U_train.size(2)
            nb_channels = contextual_ds.U_train.size(1)
            # if args_vision.model_name == 'SpatialAttn':
            #     script = importlib.import_module(f"pipeline.dl_models.SpatialAttn.{args_vision.model_name}.load_config")
            #     importlib.reload(script) 
            #     config_vision =script.args# script.get_config(C_netmob)
            #     dic_config_vision = vars(config_vision)
            #     dic_config_vision.update(vars(args_vision))
            #     args_vision = Namespace(**dic_config_vision)
            script = importlib.import_module(f"pipeline.dl_models.vision_models.{args_vision.model_name}.load_config")
            importlib.reload(script) 
            config_vision =script.get_config(input_size,nb_channels)# script.get_config(C_netmob)
            args_vision = Namespace(**{**vars(config_vision),**vars(args_vision)})

        else: 
            raise NotImplementedError(f"args_vision.dataset_name '{args_vision.dataset_name}' n'est probabelment pas importé correctement. Modifier le code.")
            C_netmob = NetMob_ds.U_train.size(2) if len(NetMob_ds.U_train.size())==6 else  NetMob_ds.U_train.size(1)# [B,N,C,H,W,L]  or [B,C,H,W,L] 
            H,W,L = NetMob_ds.U_train.size(-3),NetMob_ds.U_train.size(-2),NetMob_ds.U_train.size(-1)
            script = importlib.import_module(f"pipeline.dl_models.vision_models.{args_vision.model_name}.load_config")
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
    remove_from_dict = []
    if len(args.contextual_dataset_names) > 0: 
        contextual_ds,args = tackle_input_data(target_ds,invalid_dates,coverage_period,args,normalize)
        for name_i,contextual_ds_i in contextual_ds.items():
            if hasattr(args,'contextual_kwargs'):
                kwargs_i = args.contextual_kwargs[name_i]
                if not 'use_only_for_common_dates' in kwargs_i.keys():
                    kwargs_i['use_only_for_common_dates'] = False
                if not 'vision_model_name' in kwargs_i.keys():
                    kwargs_i['vision_model_name'] = None
                

            # --- Just to correspond with old version: 
            else:
                kwargs_i = {'use_only_for_common_dates':False,
                            'vision_model_name':args.vision_model_name  if hasattr(args,'vision_model_name') else None,
                            'stacked_contextual':args.stacked_contextual,
                            'need_global_attn':args.need_global_attn if hasattr(args,'need_global_attn') else None,
                            'vision_input_type':args.vision_input_type if hasattr(args,'vision_input_type') else None,
                            'vision_model_name': args.vision_model_name if hasattr(args,'vision_model_name') else None,
                            }
                if 'netmob_POIs' in name_i:
                    kwargs_i['NetMob_selected_apps'] = args.NetMob_selected_apps
                    kwargs_i['NetMob_transfer_mode'] = args.NetMob_transfer_mode
                    kwargs_i['NetMob_selected_tags'] = args.NetMob_selected_tags
                    
                if not hasattr(args,'contextual_kwargs'):
                    args.contextual_kwargs = {}
                args.contextual_kwargs[name_i] = kwargs_i
            # --- ...

            if kwargs_i['use_only_for_common_dates']:
                remove_from_dict.append(name_i)

            else:
                # If the contextual dataset does not need a feature extractor model:
                if kwargs_i['vision_model_name'] is None: 
                    kwargs_i['args_vision'] = argparse.ArgumentParser(description='args_vision').parse_args(args=[])

                    ## Get the new added dimension: 
                    #    Case 2.i:   in case we compute node attributes with attention or in case 'per_station' is in the name of the contextual dataset:
                    weather_in_backbone = ('weather' in args.contextual_kwargs.keys()) and ('emb_dim' in args.contextual_kwargs['weather'].keys()) and ('backbone_model' in kwargs_i.keys()) and kwargs_i['backbone_model']
                    if (kwargs_i['need_global_attn']) or ('per_station' in name_i): 
                        print('  name_i:',name_i)
                        if 'latent_dim' in kwargs_i['attn_kwargs'].keys() :
                            latent_dim = kwargs_i['attn_kwargs']['latent_dim'] 
                            
                        # If Spatial Attention inspired from STAEformer: 
                        elif 'simple_embedding_dim' in kwargs_i['attn_kwargs'].keys() and (kwargs_i['attn_kwargs']['simple_embedding_dim'] > 0):
                            latent_dim = kwargs_i['attn_kwargs']['simple_embedding_dim']
                        else:
                            latent_dim = ( 
                                          (kwargs_i['attn_kwargs']['input_embedding_dim'] if ('init_adaptive_query_dim' not in kwargs_i['attn_kwargs'].keys() or kwargs_i['attn_kwargs']['init_adaptive_query_dim'] == 0) else kwargs_i['attn_kwargs']['init_adaptive_query_dim'])
                                        + (kwargs_i['attn_kwargs']['adaptive_embedding_dim'] if 'adaptive_embedding_dim' in kwargs_i['attn_kwargs'].keys() else 0)
                                        + (kwargs_i['attn_kwargs']['tod_embedding_dim'] if 'tod_embedding_dim' in kwargs_i['attn_kwargs'].keys() else 0)
                                        + (kwargs_i['attn_kwargs']['dow_embedding_dim'] if 'dow_embedding_dim' in kwargs_i['attn_kwargs'].keys() else 0)
                                        + (args.contextual_kwargs['weather']['emb_dim'] if weather_in_backbone else 0)
                            )
                      
                    #     Case 2.ii:
                    else:
                        latent_dim = 1

                    if ('netmob_POIs' in name_i) and (not kwargs_i['need_global_attn']):
                        add_dim = len(kwargs_i['NetMob_selected_apps'])*len(kwargs_i['NetMob_transfer_mode'])*len(kwargs_i['NetMob_selected_tags']) 
                    else:
                        # add_dim =  latent_dim*contextual_ds_i.C 
                        add_dim =  latent_dim
                    args.contextual_kwargs[name_i]['added_dim'] = add_dim


                    ##  Case 1: We don't stack contextual Data
                    if not kwargs_i['stacked_contextual']:
                        args.contextual_kwargs[name_i]['out_dim'] = add_dim
                        # print(f"   Contextual dataset '{name_i}' out_dim: {args.contextual_kwargs[name_i]['out_dim']}")
                    
                    ##  Case 2: We stack contextual Data
                    else:
                        args.C = args.C + add_dim

                # In case the contextual daatset need a feature extractor model: 
                else:
                    if kwargs_i['stacked_contextual']:
                        raise ValueError(f"You defined a feature extractor model from your contextual data {name_i} but you plan to stack the contextual in a channel.\n\
                                        It's not consistent as with 'stacked_contextual' you are not supposed to extract feature information before the core model.\n\
                                        Otherwise, set 'stacked_contextual' to False\
                                        ")
                    # else:
                    #     # print('   vision_input_type', kwargs_i['vision_input_type'])
                    #     # print('   vision_model_name', kwargs_i['vision_model_name'])
                    #     # args_vision = Namespace(**{'dataset_name': name_i, 'model_name':kwargs_i['vision_model_name'],'input_type':kwargs_i['vision_input_type']})

                    #     # args_vision = tackle_config_of_feature_extractor_module(contextual_ds,args_vision)
                    #     # kwargs_i['args_vision'] = args_vision
                    #     args_vision = Namespace(**kwargs_i['attn_kwargs'])

        for name_i in remove_from_dict:
            del contextual_ds[name_i]
        args.contextual_dataset_names =  [name_i for name_i in args.contextual_dataset_names if name_i not in remove_from_dict]
        print('\nSize of Contextual datasets:')
        print(f"   Init Dataset: '{[c_i.raw_values.size()for _,c_i in contextual_ds.items()]}")
        nan_values = [torch.isnan(c_i.raw_values).sum().item() for _,c_i in contextual_ds.items()]
        if sum(nan_values) > 0:
            print(f"{nan_values} Nan values")
        print('   TRAIN contextual_ds:',[c_i.U_train.size() for _,c_i in contextual_ds.items()])
        print('   VALID contextual_ds:',[c_i.U_valid.size() for  _,c_i in contextual_ds.items()]) if hasattr(target_ds,'U_valid') else None
        print('   TEST contextual_ds:',[c_i.U_test.size() for  _,c_i in contextual_ds.items()]) if hasattr(target_ds,'U_test') else None
    else: 
        contextual_ds = {}

    return args,contextual_ds