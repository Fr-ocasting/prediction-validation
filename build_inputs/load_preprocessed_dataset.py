# Relative path:
import sys 
import os 
import importlib
import pandas as pd 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal inputs:
from build_inputs.load_contextual_data import tackle_contextual
from build_inputs.load_datasets_to_predict import load_datasets_to_predict,get_intersect_of_coverage_periods
from build_inputs.load_calendar import load_calendar,update_args_embedding
from utils.utilities import filter_args
from utils.utilities import get_time_step_per_hour
from dataset import DataSet
from dataset import PersonnalInput
from utils.seasonal_decomposition import fill_and_decompose_df


def update_contextual_tensor(dataset_name,args,need_local_spatial_attn,ds_to_predict,contextual_tensors,contextual_ds,ds_which_need_spatial_attn_per_station,contextual_positions,dict_pos_node_attr2ds,ds_which_need_global_attn):
    
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

    
    contextual_positions[dataset_name] = pos_contextual_i

    if need_local_spatial_attn:
        ds_which_need_spatial_attn_per_station.append(dataset_name)
        setattr(args,f"pos_{dataset_name}",pos_contextual_i)
    else:
        dict_pos_node_attr2ds[pos_contextual_i] = dataset_name
        if args.contextual_kwargs[dataset_name]['need_global_attn']:
            ds_which_need_global_attn.append(dataset_name)

    
    return contextual_tensors,ds_to_predict,ds_which_need_spatial_attn_per_station,contextual_positions,dict_pos_node_attr2ds,ds_which_need_global_attn


def add_contextual_data(args,target_ds,contextual_ds,dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test):
    """
    We have to add in 'dict_pos_node_attr2ds', all the position of tensor which are directly related to a node information 
    >>> ex: subway-out which has 40 Nodes, and each of them is directly related to an information in the graph 

    We have to add f"pos_{ds_name_i}" a list of N tensor positions, for each dataset  'ds_name_i' which is inside 'ds_which_need_spatial_attn_per_station'.

    ds_which_need_spatial_attn_per_station: List of dataset names which need sub-attention module at each spatial unit.
        >>> ex:  Selection of Pi pois around a station i  and then attention to reduce the Pi time serie to a unique channel.


    ds_which_need_global_attn:  List of dataset names which need a spatial attention
        >>> ex : Selection of P spatiaal units and then attention to reduce the P time-series to a N*Cp tmie-serie. 
    """
    # === Define DataLoader : 
    ds_which_need_spatial_attn_per_station = []
    dict_pos_node_attr2ds = {}
    ds_which_need_global_attn = []
    target_ds.normalizers = {target_ds.target_data:target_ds.normalizer}



    # Create 'contextual_tensors' and 'contextual_positions' and Add calendar data to the contextual tensors:
    contextual_tensors = {f'calendar_{calendar_type}': {'train': dict_calendar_U_train[calendar_type],
                                'valid': dict_calendar_U_valid[calendar_type],
                                'test': dict_calendar_U_test[calendar_type]} for calendar_type in dict_calendar_U_train.keys()
                                }
    contextual_positions= {f'calendar_{calendar_type}': list(contextual_tensors.keys()).index(f'calendar_{calendar_type}') for calendar_type in dict_calendar_U_train.keys()}
    # ...

    # == Define contextual tensor for Calibration : 
    # pos_calibration_calendar = list(contextual_tensors.keys()).index(f'calendar_{args.calibration_calendar_class}')
    # contextual_positions['calibration_calendar'] = pos_calibration_calendar
    # ==
    module_path = f"load_inputs.{args.target_data}"
    module = importlib.import_module(module_path)
    USELESS_DATES = module.USELESS_DATES
    for dataset_name in args.contextual_dataset_names:
        contextual_ds_i = contextual_ds[dataset_name]

            
        if dataset_name in ['netmob_image_per_station','netmob_bidon','netmob_video_lyon']:
             need_local_spatial_attn = False
             contextual_tensors,target_ds,ds_which_need_spatial_attn_per_station,contextual_positions,dict_pos_node_attr2ds,ds_which_need_global_attn = update_contextual_tensor(dataset_name,args,need_local_spatial_attn,
                                                                                                                             target_ds,contextual_tensors,contextual_ds_i,
                                                                                                                             ds_which_need_spatial_attn_per_station,contextual_positions,
                                                                                                                             dict_pos_node_attr2ds,ds_which_need_global_attn)
             

        elif dataset_name in ['subway_out','subway_in','subway_indiv','PeMS08_flow','PeMS08_occupancy','PeMS08_speed']:
            need_local_spatial_attn = False
            contextual_tensors,target_ds,ds_which_need_spatial_attn_per_station,contextual_positions,dict_pos_node_attr2ds,ds_which_need_global_attn = update_contextual_tensor(dataset_name,args,need_local_spatial_attn,
                                                                                                                             target_ds,contextual_tensors,contextual_ds_i,
                                                                                                                             ds_which_need_spatial_attn_per_station,contextual_positions,
                                                                                                                             dict_pos_node_attr2ds,ds_which_need_global_attn) 
            if args.data_augmentation and args.DA_method == 'noise':
                if args.DA_noise_from == 'MSTL':
                    decomposition = fill_and_decompose_df(contextual_ds_i.raw_values,
                                                        contextual_ds_i.tensor_limits_keeper.df_verif_train,
                                                        contextual_ds_i.time_step_per_hour,
                                                        contextual_ds_i.spatial_unit,
                                                        min_count = args.DA_min_count, 
                                                        periods = contextual_ds_i.periods,
                                                        dataset_name =dataset_name)
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
        elif dataset_name in ['netmob_POIs','bike_in','bike_out']:
            need_local_spatial_attn = False
            contextual_tensors,target_ds,ds_which_need_spatial_attn_per_station,contextual_positions,dict_pos_node_attr2ds,ds_which_need_global_attn = update_contextual_tensor(dataset_name,args,need_local_spatial_attn,
                                                                                                                             target_ds,contextual_tensors,contextual_ds_i,
                                                                                                                             ds_which_need_spatial_attn_per_station,contextual_positions,
                                                                                                                             dict_pos_node_attr2ds,ds_which_need_global_attn)
            if args.data_augmentation and args.DA_method == 'noise':
                raise NotImplementedError('Pas implémenté" encore. Copier le build Noise de subway_out ?')     

                
        elif dataset_name == 'netmob_POIs_per_station':
            need_local_spatial_attn = True
            contextual_tensors,target_ds,ds_which_need_spatial_attn_per_station,contextual_positions,dict_pos_node_attr2ds,ds_which_need_global_attn = update_contextual_tensor(dataset_name,args,need_local_spatial_attn,
                                                                                                                             target_ds,contextual_tensors,contextual_ds_i,
                                                                                                                             ds_which_need_spatial_attn_per_station,contextual_positions,
                                                                                                                             dict_pos_node_attr2ds,ds_which_need_global_attn)
            if args.data_augmentation and args.DA_method == 'noise':
                 raise NotImplementedError('Pas implémenté" encore. Copier le build Noise de subway_out')


        elif dataset_name == 'subway_out_per_station':
            need_local_spatial_attn = True
            contextual_tensors,target_ds,ds_which_need_spatial_attn_per_station,contextual_positions,dict_pos_node_attr2ds,ds_which_need_global_attn = update_contextual_tensor(dataset_name,args,need_local_spatial_attn,
                                                                                                                             target_ds,contextual_tensors,contextual_ds_i,
                                                                                                                             ds_which_need_spatial_attn_per_station,contextual_positions,
                                                                                                                             dict_pos_node_attr2ds,ds_which_need_global_attn)

             # build Noises :
            if args.data_augmentation and args.DA_method == 'noise':
                if args.DA_noise_from == 'MSTL':
                    decomposition = fill_and_decompose_df(contextual_ds_i[0].raw_values,
                                                        contextual_ds_i[0].tensor_limits_keeper.df_verif_train,
                                                        contextual_ds_i[0].time_step_per_hour,
                                                        contextual_ds_i[0].spatial_unit,
                                                        min_count = args.DA_min_count, 
                                                        periods = contextual_ds_i[0].periods,
                                                        dataset_name = dataset_name)
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
    target_ds.contextual_positions = contextual_positions
    args.contextual_positions = contextual_positions
    args.ds_which_need_spatial_attn_per_station = ds_which_need_spatial_attn_per_station
    args.dict_pos_node_attr2ds = dict_pos_node_attr2ds
    args.ds_which_need_global_attn = ds_which_need_global_attn
 
    return(target_ds,args)


def load_input_and_preprocess(dims,normalize,invalid_dates,args,data_T,coverage_period,name,minmaxnorm,standardize,freq = None,step_ahead = None,horizon_step =None, tensor_limits_keeper=None):
    df_dates = pd.DataFrame(coverage_period)
    df_dates.columns = ['date']
    args_DataSet = filter_args(DataSet, args,excluded_args = ['step_ahead','time_step_per_hour','horizon_step','minmaxnorm','standardize'])

    preprocessed_ds = PersonnalInput(invalid_dates,args,name = name, tensor = data_T, dates = df_dates,

                           dims =dims,
                           step_ahead = step_ahead if step_ahead is not None else args.step_ahead,
                           time_step_per_hour = get_time_step_per_hour(freq) if freq is not None else get_time_step_per_hour(args.freq),
                           horizon_step = horizon_step if horizon_step is not None else args.horizon_step,
                           minmaxnorm = minmaxnorm,
                           standardize = standardize,

                           **args_DataSet)
    
    preprocessed_ds.preprocess(args.train_prop,args.valid_prop,args.test_prop,args.train_valid_test_split_method,normalize)

    return preprocessed_ds


def load_complete_ds(args,coverage_period = None,normalize = True):

    union_invalid_dates,intersect_coverage_period =get_intersect_of_coverage_periods(args,coverage_period)
    target_ds= load_datasets_to_predict(args,
                                        invalid_dates = union_invalid_dates,
                                        coverage_period = intersect_coverage_period,
                                        normalize=normalize)
    
    # Calendar data for Calibration : 
    dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test = load_calendar(target_ds)
    args = update_args_embedding(args,dict_calendar_U_train)

    # Contextual: 
    args,contextual_ds = tackle_contextual(target_ds,
                                           invalid_dates= union_invalid_dates,
                                           coverage_period=intersect_coverage_period,
                                           args=args,
                                           normalize = normalize)


    # Add Contextual Tensors and their contextual_positions: 
    target_ds,args = add_contextual_data(args,target_ds,contextual_ds,dict_calendar_U_train,dict_calendar_U_valid,dict_calendar_U_test)
    # Update/Set arguments: 
    assert target_ds.U_train.dim() == 3, f'Feature Vector does not have the good dimension. Expected shape dimension [B,N,L], got {target_ds.U_train.dim()} dim: {target_ds.U_train.size()}'
    return(target_ds,contextual_ds,args) #,args.dic_class2rpz)