# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal inputs:
from build_inputs.load_contextual_data import tackle_contextual
from build_inputs.load_datasets_to_predict import load_datasets_to_predict
from build_inputs.load_calendar import load_calendar,get_args_embedding
from utils.utilities import filter_args
from utils.utilities import get_time_step_per_hour
from dataset import DataSet
from dataset import PersonnalInput

# from build_inputs.load_calendar import tackle_calendar
from utils.seasonal_decomposition import fill_and_decompose_df
from constants.paths import USELESS_DATES
import pandas as pd 


def update_contextual_tensor(dataset_name,args,need_local_spatial_attn,ds_to_predict,contextual_tensors,contextual_ds,ds_which_need_spatial_attn,positions,pos_node_attributes,dict_node_attr2dataset,node_attr_which_need_attn):
    
    if type(contextual_ds) == list:
        contextual_tensors.update({f'{dataset_name}_{k}': {'train': ds_i.U_train,
                                    'valid': ds_i.U_valid if hasattr(ds_i,'U_valid') else None,
                                    'test': ds_i.U_test  if hasattr(ds_i,'U_test') else None}
                                    for k,ds_i in enumerate(contextual_ds)
                                    }
                                    )
        pos_contextual_i = [list(contextual_tensors.keys()).index(f'{dataset_name}_{k}') for k,ds_i in enumerate(contextual_ds)]
        ds_to_predict.normalizers.update({dataset_name:contextual_ds[0].normalizer})
        for k,ds_i in enumerate(contextual_ds):
            setattr(args,f"n_units_{dataset_name}_{k}",ds_i.U_train.size(1))
            setattr(args,f"input_dim_{dataset_name}_{k}",ds_i.U_train.size(2))
    else:
        contextual_tensors.update({dataset_name: {'train': contextual_ds.U_train,
                            'valid': contextual_ds.U_valid if hasattr(contextual_ds,'U_valid') else None,
                            'test': contextual_ds.U_test  if hasattr(contextual_ds,'U_test') else None}
                            }
                            )
        pos_contextual_i = list(contextual_tensors.keys()).index(dataset_name)
        ds_to_predict.normalizers.update({dataset_name:contextual_ds.normalizer})
        setattr(args,f"n_units_{dataset_name}",contextual_ds.U_train.size(1))
        setattr(args,f"input_dim_{dataset_name}",contextual_ds.U_train.size(2))

    
    positions[dataset_name] = pos_contextual_i

    if need_local_spatial_attn:
        ds_which_need_spatial_attn.append(dataset_name)
        setattr(args,f"pos_{dataset_name}",pos_contextual_i)
    else:
        pos_node_attributes.append(pos_contextual_i)
        dict_node_attr2dataset[pos_contextual_i] = dataset_name
        if args.compute_node_attr_with_attn:
            node_attr_which_need_attn.append(dataset_name)

    
    return contextual_tensors,ds_to_predict,ds_which_need_spatial_attn,positions,pos_node_attributes,dict_node_attr2dataset,node_attr_which_need_attn


def add_contextual_data(args,target_ds,contextual_ds,dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test):
    '''
    We have to add in 'pos_node_attributes', all the position of tensor which are directly related to a node information 
    >>> ex: subway-out which has 40 Nodes, and each of them is directly related to an information in the graph 

    We have to add in 'ds_which_need_spatial_attn', all the dataset which need a spatial aggregation before being add as a new channel.
    >>> ex : Selection of P pois around a station and then attention to reduce the P time serie to a unique channel. 

    We have to add f"pos_{ds_name_i}" a list of N tensor positions, for each dataset  'ds_name_i' which is inside 'ds_which_need_spatial_attn'.
    '''
    # === Define DataLoader : 
    ds_which_need_spatial_attn = []
    pos_node_attributes = []
    dict_node_attr2dataset = {}
    node_attr_which_need_attn = []
    target_ds.normalizers = {target_ds.target_data:target_ds.normalizer}



    # Create 'contextual_tensors' and 'positions' and Add calendar data to the contextual tensors:
    contextual_tensors = {f'calendar_{calendar_type}': {'train': dict_calendar_U_train[calendar_type],
                                'valid': dict_calendar_U_valid[calendar_type],
                                'test': dict_calendar_U_test[calendar_type]} for calendar_type in dict_calendar_U_train.keys()
                                }
    pos_calendar = [list(contextual_tensors.keys()).index(f'calendar_{calendar_type}') for calendar_type in dict_calendar_U_train.keys()]
    positions= {'calendar': pos_calendar}
    # ...

    # == Define contextual tensor for Calibration : 
    # pos_calibration_calendar = list(contextual_tensors.keys()).index(f'calendar_{args.calibration_calendar_class}')
    # positions['calibration_calendar'] = pos_calibration_calendar
    # ==


    for dataset_name in args.contextual_dataset_names:
        contextual_ds_i = contextual_ds[dataset_name]

            
        if (dataset_name == 'netmob_image_per_station') or (dataset_name == 'netmob_bidon') or (dataset_name == 'netmob_video_lyon'):
             need_local_spatial_attn = False
             contextual_tensors,target_ds,ds_which_need_spatial_attn,positions,pos_node_attributes,dict_node_attr2dataset,node_attr_which_need_attn = update_contextual_tensor(dataset_name,args,need_local_spatial_attn,
                                                                                                                             target_ds,contextual_tensors,contextual_ds_i,
                                                                                                                             ds_which_need_spatial_attn,positions,pos_node_attributes,
                                                                                                                             dict_node_attr2dataset,node_attr_which_need_attn)
             

        elif (dataset_name == 'subway_out') or (dataset_name == 'subway_in') or (dataset_name == 'subway_indiv'):
            need_local_spatial_attn = False
            contextual_tensors,target_ds,ds_which_need_spatial_attn,positions,pos_node_attributes,dict_node_attr2dataset,node_attr_which_need_attn = update_contextual_tensor(dataset_name,args,need_local_spatial_attn,
                                                                                                                             target_ds,contextual_tensors,contextual_ds_i,
                                                                                                                             ds_which_need_spatial_attn,positions,pos_node_attributes,
                                                                                                                             dict_node_attr2dataset,node_attr_which_need_attn) 
            if args.data_augmentation and args.DA_method == 'noise':
                if args.DA_noise_from == 'MSTL':
                    decomposition = fill_and_decompose_df(contextual_ds_i.raw_values,
                                                        contextual_ds_i.tensor_limits_keeper.df_verif_train,
                                                        contextual_ds_i.time_step_per_hour,
                                                        contextual_ds_i.spatial_unit,
                                                        min_count = args.DA_min_count, 
                                                        periods = contextual_ds_i.periods)
                    df_noises = pd.DataFrame({col : decomposition[col]['resid'] for col in decomposition.keys()})
                    df_noises = df_noises[contextual_ds_i.spatial_unit]
                elif args.DA_noise_from == 'Homogenous':
                    df_verif_train = contextual_ds_i.tensor_limits_keeper.df_verif_train
                    dates_used_in_train = pd.Series(pd.concat([df_verif_train[c] for c in df_verif_train.columns]).unique()).sort_values() 
                    reindex_dates = pd.date_range(dates_used_in_train.min(),dates_used_in_train.max(),freq=f"{1/contextual_ds_i.time_step_per_hour}h")
                    reindex_dates = reindex_dates[~reindex_dates.hour.isin(USELESS_DATES['hour'])&~reindex_dates.hour.isin(USELESS_DATES['weekday'])]
                    df_noises = pd.DataFrame({col : [1]*len(reindex_dates) for col in contextual_ds_i.spatial_unit},index =reindex_dates )
                else :
                    raise NotImplementedError(f"Noise from {args.DA_noise_from} has not been implemented")
                
                target_ds.noises[dataset_name] = df_noises         
        elif (dataset_name == 'netmob_POIs'):
            need_local_spatial_attn = False
            contextual_tensors,target_ds,ds_which_need_spatial_attn,positions,pos_node_attributes,dict_node_attr2dataset,node_attr_which_need_attn = update_contextual_tensor(dataset_name,args,need_local_spatial_attn,
                                                                                                                             target_ds,contextual_tensors,contextual_ds_i,
                                                                                                                             ds_which_need_spatial_attn,positions,pos_node_attributes,
                                                                                                                             dict_node_attr2dataset,node_attr_which_need_attn)
            if args.data_augmentation and args.DA_method == 'noise':
                raise NotImplementedError('Pas implémenté" encore. Copier le build Noise de subway_out ?')     

                
        elif dataset_name == 'netmob_POIs_per_station':
            need_local_spatial_attn = True
            contextual_tensors,target_ds,ds_which_need_spatial_attn,positions,pos_node_attributes,dict_node_attr2dataset,node_attr_which_need_attn = update_contextual_tensor(dataset_name,args,need_local_spatial_attn,
                                                                                                                             target_ds,contextual_tensors,contextual_ds_i,
                                                                                                                             ds_which_need_spatial_attn,positions,pos_node_attributes,
                                                                                                                             dict_node_attr2dataset,node_attr_which_need_attn)
            if args.data_augmentation and args.DA_method == 'noise':
                 raise NotImplementedError('Pas implémenté" encore. Copier le build Noise de subway_out')


        elif dataset_name == 'subway_out_per_station':
            need_local_spatial_attn = True
            contextual_tensors,target_ds,ds_which_need_spatial_attn,positions,pos_node_attributes,dict_node_attr2dataset,node_attr_which_need_attn = update_contextual_tensor(dataset_name,args,need_local_spatial_attn,
                                                                                                                             target_ds,contextual_tensors,contextual_ds_i,
                                                                                                                             ds_which_need_spatial_attn,positions,pos_node_attributes,
                                                                                                                             dict_node_attr2dataset,node_attr_which_need_attn)

             # build Noises :
            if args.data_augmentation and args.DA_method == 'noise':
                if args.DA_noise_from == 'MSTL':
                    decomposition = fill_and_decompose_df(contextual_ds_i[0].raw_values,
                                                        contextual_ds_i[0].tensor_limits_keeper.df_verif_train,
                                                        contextual_ds_i[0].time_step_per_hour,
                                                        contextual_ds_i[0].spatial_unit,
                                                        min_count = args.DA_min_count, 
                                                        periods = contextual_ds_i[0].periods)
                    df_noises = pd.DataFrame({col : decomposition[col]['resid'] for col in decomposition.keys()})
                    df_noises = df_noises[contextual_ds_i[0].spatial_unit]
                elif args.DA_noise_from == 'Homogenous':
                    df_verif_train = contextual_ds_i[0].tensor_limits_keeper.df_verif_train
                    dates_used_in_train = pd.Series(pd.concat([df_verif_train[c] for c in df_verif_train.columns]).unique()).sort_values() 
                    reindex_dates = pd.date_range(dates_used_in_train.min(),dates_used_in_train.max(),freq=f"{1/contextual_ds_i[0].time_step_per_hour}h")
                    reindex_dates = reindex_dates[~reindex_dates.hour.isin(USELESS_DATES['hour'])&~reindex_dates.hour.isin(USELESS_DATES['weekday'])]
                    df_noises = pd.DataFrame({col : [1]*len(reindex_dates) for col in contextual_ds_i[0].spatial_unit},index =reindex_dates )
                else :
                    raise NotImplementedError(f"Noise from {args.DA_noise_from} has not been implemented")


                target_ds.noises[dataset_name] = df_noises
        else:
            raise NotImplementedError(f'Dataset {dataset_name} has not been implemented')
        

    target_ds.contextual_tensors = contextual_tensors
    target_ds.get_dataloader()

    # Maybe useless to send it to the both 
    target_ds.contextual_positions = positions
    args.contextual_positions = positions
    args.ds_which_need_spatial_attn = ds_which_need_spatial_attn
    args.pos_node_attributes = pos_node_attributes
    args.dict_node_attr2dataset = dict_node_attr2dataset
    args.node_attr_which_need_attn = node_attr_which_need_attn
 
    return(target_ds,args)


def load_input_and_preprocess(dims,normalize,invalid_dates,args,data_T,coverage_period):
    df_dates = pd.DataFrame(coverage_period)
    df_dates.columns = ['date']
    args_DataSet = filter_args(DataSet, args)

    preprocessed_ds = PersonnalInput(invalid_dates,args, tensor = data_T, dates = df_dates,
                            time_step_per_hour = get_time_step_per_hour(args.freq),
                            #minmaxnorm = dataset.minmaxnorm,
                            #standardize = dataset.standardize,
                           dims =dims,
                           **args_DataSet)
    
    preprocessed_ds.preprocess(args.train_prop,args.valid_prop,args.test_prop,args.train_valid_test_split_method,normalize)

    return preprocessed_ds


def load_complete_ds(args,coverage_period = None,normalize = True):
    # Load subway-in DataSet:
    target_ds,invalid_dates,intersect_coverage_period = load_datasets_to_predict(args,coverage_period,normalize)
    
    # Calendar data for Calibration : 
    dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test = load_calendar(target_ds)
    args = get_args_embedding(args,dict_calendar_U_train)

    # Contextual: 
    args,contextual_ds = tackle_contextual(target_ds,invalid_dates,intersect_coverage_period,args,normalize = normalize)

    # Add Contextual Tensors and their positions: 
    target_ds,args = add_contextual_data(args,target_ds,contextual_ds,dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test)
    # Update/Set arguments: 
    assert target_ds.U_train.dim() == 3, f'Feature Vector does not have the good dimension. Expected shape dimension [B,N,L], got {target_ds.U_train.dim()} dim: {target_ds.U_train.size()}'
    return(target_ds,contextual_ds,args) #,args.dic_class2rpz)