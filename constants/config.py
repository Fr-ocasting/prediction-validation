import torch 
import argparse
import random
import os 
import importlib
from argparse import Namespace
import sys 
import os 
import gc
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)


def get_config(model_name,dataset_names,dataset_for_coverage,config = {}):
    config['model_name'] = model_name
    config['dataset_names'] = dataset_names
    config['dataset_for_coverage'] = dataset_for_coverage

    config['bool_sanity_checker'] = False
    config['grad_clipping_norm'] = None  # 1.0 by default if grad clipping. 

    # Contextual Information: Calendar
    if 'calendar' in dataset_names:
        config['calendar_types'] = ['dayofweek','timeofday']  # IF 'calendar' in 'dataset_names': Define the type of calendar information used in the model 
    if 'calendar_embedding' in dataset_names:
        config['embedding_calendar_types'] = ['dayofweek', 'hour','minute']  # IF 'calendar_embedding' in ['dayofweek', 'hour', 'minute', 'bank_holidays', 'school_holidays', 'remaining_holidays']

    #importlib.reload(data_module)
    #config['num_nodes'] = data_module.num_nodes
    #config['C'] =  data_module.C
    
    # === Common config for everyone: ===
    config['device'] =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['optimizer'] = 'adamw' #['sgd','≈','adamw']
    config['single_station']= False
    config['loss_function_type'] = 'HuberLoss' # 'HuberLoss' # 'MSE' #'quantile' # 'masked_mae' 
    config['freq'] = '15min'

    config['minmaxnorm'] = True   # If True: apply MinMax Normalisation on each time-series
    config['standardize'] = False  # If True: apply Zscore Normalisation on each time-series

    config['learnable_adj_matrix'] = False
    config['learnable_adj_top_k'] = 8  # If learnable_adj_matrix is True, set the number of kept node links on the graph conv
    config['learnable_adj_embd_dim'] = 8 # If learnable_adj_matrix is True, set the embedding dim of each nodes

    config['contextual_positions'] = {}
    config['quick_vision'] = False #if True then load small NetMob tensor with torch.randn(), instead of big one with pickle.load() and torch concat
    config['evaluate_complete_ds'] = False # True  # -> Compute an extra training through the entire complete dataset (with init train/valid/test split)
    config['train_valid_test_split_method'] =  'similar_length_method' # 'iterative_method' #'similar_length_method'
    config['set_spatial_units'] =  None # ['BON','SOI','GER','CHA']  # None -> Select a sub-set of desired spatial-units to work on.
    config['hp_tuning_on_first_fold'] = False # If True then we remove the first fold from the K-fold validation, considering it has been used for HP tuning.
    config['keep_best_weights'] = False # If True then keep models weights associated to the best valid loss obtained. get best weights from trainer.best_weights
    # Optimization 
    if torch.cuda.is_available():
        config['num_workers'] = 2 # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
        config['persistent_workers'] = True # False 
        config['pin_memory'] = True # False 
        config['prefetch_factor'] = 2 # None, 2,3,4,5 ... 
        config['drop_last'] = False  # True
        config['mixed_precision'] = True # True # False
    else:
        config['num_workers'] = 0 # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
        config['persistent_workers'] = False # False 
        config['pin_memory'] = True # False 
        config['prefetch_factor'] = None # None, 2,3,4,5 ... 
        config['drop_last'] = False  # True      
        config['mixed_precision'] = False # True # False
    
    config['non_blocking'] = True
    config['torch_compile'] = False # 'compile', 'jit_scirpt',  None, False 
    config['backend'] ='inductor' #'inductor' #'cudagraphs'
    config['prefetch_all'] = False

    # ...

    # === Denoising ===
    config['denoising_names'] = [] #If empty, then no denoising. Else, apply denoising on dataset_names in list (['subway_in', 'subway_out', .... ,])
    config['denoiser_names'] = ["median"]   # Apply chaining denoising. Default: ``["median"]``. choices: ['median','savitzky_golay_causal','exponential']
    config['denoising_modes'] = ["train"]             # Apply denoisin on a specific datasets (train, valid, test)
    config['denoiser_kwargs'] = {"median": {"kernel_size": 5}}  # Apply denoising function with specific config .


    # === Contextual Data Kwargs:
    config['contextual_kwargs'] = {'netmob_POIs': {'need_global_attn':True, 
                                                   'stacked_contextual': True,
                                                   'NetMob_selected_apps' : [],
                                                    'NetMob_transfer_mode' :  ['DL'], #,'UL'] # ['DL'] # ['UL'] #['DL','UL']
                                                    'NetMob_selected_tags' : ['iris'],#['iris','stadium','station','university']#['park','stadium','university','station','shop','nightclub','parkings','theatre','iris','transit','public_transport']
                                                    'NetMob_expanded' : '', # '' # '_expanded'
                                                    'NetMob_only_epsilon': False, # if True then look at NetMob data in InputsEpsilon instead of Input:  '/POIs/netmob_POI_Lyon{args.NetMob_expanded}/InputsEpsilon/{id_station}'
                                        },
                                    'subway_out': {'need_global_attn':False, 
                                                   'stacked_contextual': True,
                                                   'model_name': None, # Define the type of model used to extract contextual information from NetMob
                                                   'input_type': None, # 'image_per_stations' # 'unique_image_through_lyon'  
                                                   'grn_out_dim': 0, # If >0 then stack a GRN layer to the output module
                                        },
                                    }

                                    
    # === Ray config ===
    config['ray'] = False # True
    config['ray_scheduler'] = 'ASHA' #None
    config['ray_search_alg'] = 'HyperOpt' #  'HyperOpt' # None
    config['grace_period'] = 2
    config['HP_max_epochs'] = 100

    # Config Quantile Calibration 
    config['alpha'] = 0.1
    config['conformity_scores_type'] = 'max_residual'   # Define the function to compute the non-conformity scores
    config['quantile_method'] =  'compute_quantile_by_class' # 'classic' Define type of method used to calcul quantile.  'classic':  Quantile through the entiere dataset  / 'weekday_hour': 
    config['calibration_calendar_class'] = 0  # Calibrates data by defined calendar-class 
    config['type_calib'] = 'classic'  # Calibration after quantile regression. If 'classic' then no calibration. If 'CQR' then CQR calibration

    # Data Augmentation:
    config['data_augmentation'] = False # If True then augment Data in Training DataSet
    config['DA_moment_to_focus'] = None  #[{'hours':[0,23],'weekdays':[1,3]}], # None
    config['DA_min_count'] = 5 #  int: Set the minimum of representant from a tuple (weekday,hour,minute) to consider its median for interpolation. If not >5, then take the median of (weekday,hour). If still not > 5, then take the median of (hour)
  
    config['DA_method'] = 'noise' # choices: ['noise','interpolation]. Set the method of Data Augmentation   
    config['DA_alpha'] = 1 # 0.2 # float: >0.  Parameter if the method is 'noise'. Is a multiplicative factor of the noise. Help to set its amplitude. 
    config['DA_prop'] = 1 #  SET = 1. SEEMS NOT WORKING  # float: between [0,1]: is the proportion of Data which will be augmented. 
    config['DA_noise_from'] = 'MSTL' # choices: ['MSTL','Homogenous'] Set where the noise amplitude is based from. MSTL is computationaly expensive. Constant set a noise = 1 for every single time-step and spatial-unit.

    # Config DataSet:
    config['H'] = 6
    config['W'] = 0
    config['D'] = 1
    config['step_ahead'] = 4
    config['horizon_step'] = 1 # If >1 then model will predict until config['step_ahead'] by horizon_step. Ex: step_ahead = 4 and horizon_step = 2, then model will step_ahead 2 and step_ahead 4.
    config['L'] = config['H']+config['W']+config['D']

    # Split proportion
    config['shuffle'] = True # True # False    # --> Can be set 'False' if we need to Visualise prediction of Training set without Shuffle 
    config['train_prop'] = 0.6
    config['calib_prop'] = None #0.5 # None     # -->  Don't set 'calib_prop = 0' otherwise bug
    config['valid_prop'] = 0.2  
    config['expanding_train'] = None # 0.1 # None  # If not None, float between ]0:1[. Set the proportion of last training data to keep for training.
    config['graph_subset'] = None # 0.1 # None  # If not None, float between ]0:1[. Random selection of a proportion of nodes to keep from the full graph
    config['test_prop'] = 1 - (config['train_prop'] + config['valid_prop']) 
    assert config['train_prop']+ config['valid_prop'] < 1.0, f"train_prop + valid_prop = {config['train_prop']+ config['valid_prop']}. No Testing set"
    config['track_pi'] = False #True

    # --- Keep track on grad through training : 
    config['track_grad_norm'] = False # True # False  # If True then track the gradient norm of the model through training.

    # Validation, K-fold
    config['validation_split_method'] = 'forward_chaining_cv'  # Choices =  ['custom_blocked_cv','forward_chaining_cv']. Design the way we split the data and compute the metrics.
    config['min_fold_size_proportion'] = 0.75 # choice: continuous variable in [0:1]. Set the size of the smallest fold (fold0) as maximum-size - minimum-size*K_fold*min_fold_size_proportion
    config['no_common_dates_between_set'] = False  #If True then a shift of dataset.shift_from_first_elmt is applied. Otherwise, some pattern could be within Training and Validation DataLoader
    config['K_fold'] = 6  # int. If 1 : classic validation (only 1 model), Else : validation with K_fold according 'config['validation']
    config['current_fold'] = 0

    # ===  

    config['abs_path'] =  ('/').join(f"{os.path.abspath(os.getcwd())}".split('/')[:-1]) + '/' # f"{os.path.abspath(os.getcwd())}/"
    config['dict_ds2path'] = {}


    # Define Output dimension: 
    if config['loss_function_type'] in ['MSE','HuberLoss','masked_mae','masked_mse','huber_loss','masked_huber_loss']: 
        out_dim_factor = 1
        metrics = ['rmse','mse','mae','mape','mase']
    elif config['loss_function_type'] == 'quantile': 
        out_dim_factor = 2
        metrics  = ['MPIW','PICP']
    else: raise NotImplementedError(f"loss function {config['loss_function_type']} has not been implemented")
    config['out_dim_factor'] = out_dim_factor
    config['metrics'] = metrics
    config['use_target_as_context'] = False
    # ...

    # Temporal Graph Transformer Encoder: 
    config['temporal_graph_transformer_encoder'] = False # if 'True' then change the temporal-conv of the output module to a TemporalGraph Transformer Encoder



    return(config)


def optimizer_specific_lr(model,args):
    if args.model_name == 'CNN':
        if args.specific_lr:
            specific_lr = [{"params": model.Tembedding.parameters(), "lr": 1e-2},
                    {"params": model.Convs.parameters(), "lr": args.lr},
                    {"params": model.Dense_outs.parameters(), "lr": args.lr}
                ]
            
    elif args.model_name == 'STGCN':
        if args.specific_lr:
            specific_lr = [{"params": model.Tembedding.parameters(), "lr": 1e-2},
                    {"params": model.st_blocks.parameters(), "lr": args.lr},
                    {"params": model.output.parameters(), "lr": args.lr}
                    ]
    else:
        raise NotImplementedError(f'A specific lr by layer has been asked but it has not been defined for the model {args.model_name}.')
    
    return(specific_lr)


def get_args(model_name,dataset_names,dataset_for_coverage):
    config = get_config(model_name,dataset_names,dataset_for_coverage)
    args = convert_into_parameters(config)

    # Load Config associated to the Model: 
    if model_name is not None:
        module_path = f"pipeline.dl_models.{args.model_name}.load_config"
        module = importlib.import_module(module_path)
        importlib.reload(module)
        locals()[f"args_{args.model_name}"] = module.args
        locals()[f"args_HP_{args.model_name}"] = module.args_HP 

        # Merge Args: 
        args = Namespace(**{**vars(args),**vars(locals()[f"args_{args.model_name}"]),**vars(locals()[f"args_HP_{args.model_name}"])})
    else:
        args.HP_max_epochs=50
        args.weight_decay=0.005
        args.batch_size=32
        args.lr=5e-3
        args.dropout=0.2
        args.epochs=100
        args.scheduler=None

    if 'calendar_embedding' in args.dataset_names:
        module_path_TE = f"pipeline.dl_models.TimeEmbedding.load_config"
        module_TE = importlib.import_module(module_path_TE)
        importlib.reload(module_TE)
        args_embedding = module_TE.args
        args.args_embedding = args_embedding
    else:
        args.args_embedding = argparse.ArgumentParser(description='TimeEmbedding').parse_args(args=[])
    return(args)


def local_get_args(model_name,args_init,dataset_names,dataset_for_coverage,modification):
    ''' Load args for tramodificationining, but also allow to '''
    # Load base args
    args = get_args(model_name,dataset_names,dataset_for_coverage)
    args.ray = False

    #  evaluation on the first fold only :
    args.evaluate_complete_ds = True  # True # False // if True, then evaluation also on the entiere ds 

    if args_init is not None:
        args.args_vision = args_init.args_vision
        args.args_embedding = args_init.args_embedding 
        args.contextual_positions = args_init.contextual_positions
        args.vision_input_type = args_init.vision_input_type

    # Modification :
    for key,value in modification.items():
        setattr(args,key,value)

        if key == 'HP_max_epochs':
            if hasattr(args,'epochs'):
                if args.epochs < value:
                    args.epochs = value
            else:
                args.epochs = value
            
        if 'TE_' in key :
            key = key.replace('TE_','')
            setattr(args.args_embedding,key,value)
    

    # update each modif
    args = update_modif(args)

    return args


def update_modif(args):
    #Update modification:
    args = update_out_dim(args)
    #...
    
    # Ray tuning function can't hundle with PyTorch multiprocessing : 
    if args.ray:
        args.num_workers = 0
        args.persistent_workers = False
        args.track_pi = False
    # ...

    # Modif about num_nodes: 
    if args.set_spatial_units is not None:
        args.num_nodes = len(args.set_spatial_units)

    # Update total sequence length:
    if not args.target_data in args.dataset_names:
        args.L = 0
    else:
        args.L = args.H+args.D+args.W


    # Remove some args that are not used in the model:
    if hasattr(args,'denoising_names') and len(args.denoising_names) == 0:
        if hasattr(args,'denoising_modes'): del args.denoising_modes
        if hasattr(args,'denoiser_kwargs'): del args.denoiser_kwargs

    if not('netmob' in args.target_data) and not (sum(['netmob' in n.lower() for n in args.dataset_names]) > 0) :
        if hasattr(args,'NetMob_selected_apps'): del args.NetMob_selected_apps
        if hasattr(args,'NetMob_transfer_mode'): del args.NetMob_transfer_mode 
        if hasattr(args,'NetMob_selected_tags'): del args.NetMob_selected_tags
        if hasattr(args,'NetMob_expanded'): del args.NetMob_expanded 
        if hasattr(args,'NetMob_only_epsilon'): del args.NetMob_only_epsilon 

    if hasattr(args,'data_augmentation') and not args.data_augmentation:
        if hasattr(args,'DA_moment_to_focus'): del args.DA_moment_to_focus
        if hasattr(args,'DA_min_count'): del args.DA_min_count
        if hasattr(args,'DA_method'): del args.DA_method
        if hasattr(args,'DA_alpha'): del args.DA_alpha
        if hasattr(args,'DA_prop'): del args.DA_prop
        if hasattr(args,'DA_noise_from'): del args.DA_noise_from

    if hasattr(args,'learnable_adj_matrix') and not args.learnable_adj_matrix:
        if hasattr(args,'learnable_adj_top_k'): del args.learnable_adj_top_k
        if hasattr(args,'learnable_adj_embd_dim'): del args.learnable_adj_embd_dim
    
    if hasattr(args,'temporal_graph_transformer_encoder') and not args.temporal_graph_transformer_encoder:
        if hasattr(args,'TGE_num_layers'): del args.TGE_num_layers
        if hasattr(args,'TGE_num_heads'): del args.TGE_num_heads
        if hasattr(args,'TGE_FC_hdim'): del args.TGE_FC_hdim

    if hasattr(args,'contextual_kwargs'):
        remove_from_dict = []
        for ds_name in args.contextual_kwargs.keys():
            if ds_name not in args.dataset_names:
                remove_from_dict.append(ds_name)
        for ds_name in remove_from_dict:
            del args.contextual_kwargs[ds_name]

    if args.calib_prop is None:
        if hasattr(args,'conformity_scores_type'): del args.conformity_scores_type
        if hasattr(args,'quantile_method'): del args.quantile_method
        if hasattr(args,'calibration_calendar_class'): del args.calibration_calendar_class
        if hasattr(args,'type_calib'): del args.type_calib
    if hasattr(args,'torch_scheduler') and not args.torch_scheduler :
        if hasattr(args,'torch_scheduler_milestone'): del args.torch_scheduler_milestone
        if hasattr(args,'torch_scheduler_gamma'): del args.torch_scheduler_gamma    
        if hasattr(args,'torch_scheduler_type'): del args.torch_scheduler_type
        if hasattr(args,'torch_scheduler_lr_start_factor'): del args.torch_scheduler_lr_start_factor
    return(args)

def modification_contextual_args(args,modification):
    for key,value in modification.items():
        if key.startswith('TE_'):
            key = key.replace('TE_','')
            setattr(args.args_embedding,key,value)
        elif key.startswith('vision_') and not key == 'vision_model_name':
            key = key.replace('vision_','')
            setattr(args.args_vision,key,value)
            if key == 'grn_out_dim':
                setattr(args.args_vision,'out_dim',value)
        else:
            setattr(args,key,value)
    return(args)


def convert_into_parameters(config):
    parser = argparse.ArgumentParser()

    for key in config.keys():
        default = config[key]
        parser.add_argument(f'--{key}', type=type(default), default=default)

    args = parser.parse_args(args=[])
    return(args)

def update_out_dim(args):
    if args.loss_function_type in ['MSE','HuberLoss','masked_mae','masked_mse','huber_loss','masked_huber_loss']: 
        args.out_dim_factor = 1
        args.alpha = None
    elif args.loss_function_type == 'quantile': 
        args.out_dim_factor = 2
        args.metrics  = ['MPIW','PICP']
    else: 
        raise NotImplementedError(f'loss function {args.loss_function_type} has not been implemented')
    
    args.out_dim = args.out_dim_factor * (args.step_ahead // args.horizon_step)
    return args 