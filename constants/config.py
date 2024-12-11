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

from constants.paths import DATA_TO_PREDICT

def get_config(model_name,dataset_names,dataset_for_coverage,config = {}):
    config['model_name'] = model_name
    config['dataset_names'] = dataset_names
    config['dataset_for_coverage'] = dataset_for_coverage

    data_module = importlib.import_module(f"load_inputs.{DATA_TO_PREDICT}")
    importlib.reload(data_module)
    config['n_vertex'] = data_module.n_vertex
    config['C'] =  data_module.C
    
    # === Common config for everyone: ===
    config['device'] =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['optimizer'] = 'adamw' #['sgd','adam','adamw']
    config['single_station']= False
    config['loss_function_type'] = 'MSE' # 'MSE' #'quantile'

    config['contextual_positions'] = {}
    config['quick_vision'] = False #if True then load small NetMob tensor with torch.randn(), instead of big one with pickle.load() and torch concat
    config['netmob_transfer_mode'] = 'DL' # 'UL' # None # -> Use as NetMob input only 'DL' or 'UL' transfer mode, or both
    config['evaluate_complete_ds'] = False # True  # -> Compute an extra training through the entire complete dataset (with init train/valid/test split)
    config['train_valid_test_split_method'] =  'similar_length_method' # 'iterative_method' #'similar_length_method'
    config['set_spatial_units'] =  None # ['BON','SOI','GER','CHA']  # None -> Select a sub-set of desired spatial-units to work on.
    config['concatenation_late'] = False # True -> If True then exracted feature are concatenated at the output of the backbone model, before passing through FC layers
    # Optimization 
    if torch.cuda.is_available():
        config['num_workers'] = 2 # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
        config['persistent_workers'] = True # False 
        config['pin_memory'] = True # False 
        config['prefetch_factor'] = 2 # None, 2,3,4,5 ... 
        config['drop_last'] = False  # True
        config['mixed_precision'] = False # True # False
    else:
        config['num_workers'] = 0 # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
        config['persistent_workers'] = False # False 
        config['pin_memory'] = True # False 
        config['prefetch_factor'] = None # None, 2,3,4,5 ... 
        config['drop_last'] = False  # True      
        config['mixed_precision'] = False # True # False
    
    config['non_blocking'] = True
    config['torch_compile'] = False #True
    config['backend'] = 'inductor' #'cudagraphs'
    config['prefetch_all'] = False
    # ...

    # === Ray config ===
    config['ray'] = False # True
    config['ray_scheduler'] = 'ASHA' #None
    config['ray_search_alg'] = None #  'HyperOpt'
    config['grace_period'] = 2
    config['HP_max_epochs'] = 100

    # Config Quantile Calibration 
    config['alpha'] = 0.1
    config['conformity_scores_type'] = 'max_residual'   # Define the function to compute the non-conformity scores
    config['quantile_method'] =  'compute_quantile_by_class' # 'classic' Define type of method used to calcul quantile.  'classic':  Quantile through the entiere dataset  / 'weekday_hour': 
    config['calibration_calendar_class'] = 0  # Calibrates data by defined calendar-class 
    config['type_calib'] = 'classic'  # Calibration after quantile regression. If 'classic' then no calibration. If 'CQR' then CQR calibration

    # Config DataSet:
    config['H'] = 6
    config['W'] = 0
    config['D'] = 1
    config['step_ahead'] = 1

    if not 'subway_in' in dataset_names:
        config['L'] = 0
    config['L'] = config['H']+config['W']+config['D']

    # Split proportion
    config['shuffle'] = True # True # False    # --> Can be set 'False' if we need to Visualise prediction of Training set without Shuffle 
    config['train_prop'] = 0.6
    config['calib_prop'] = None #0.5 # None     # -->  Don't set 'calib_prop = 0' otherwise bug
    config['valid_prop'] = 0.2  
    config['test_prop'] = 1 - (config['train_prop'] + config['valid_prop']) 
    assert config['train_prop']+ config['valid_prop'] < 1.0, f"train_prop + valid_prop = {config['train_prop']+ config['valid_prop']}. No Testing set"
    config['track_pi'] = False #True


    # Validation, K-fold
    config['validation'] = 'sliding_window'  # classic / sliding_window / 
    config['no_common_dates_between_set'] = False  #If True then a shift of dataset.shift_from_first_elmt is applied. Otherwise, some pattern could be within Training and Validation DataLoader
    config['K_fold'] = 6  # int. If 1 : classic validation (only 1 model), Else : validation with K_fold according 'config['validation']
    config['current_fold'] = 0

    # ===   ===
    config['abs_path'] =  ('/').join(f"{os.path.abspath(os.getcwd())}".split('/')[:-1]) + '/' # f"{os.path.abspath(os.getcwd())}/"


    # Define Output dimension: 
    if config['loss_function_type'] == 'MSE': out_dim = 1
    elif config['loss_function_type'] == 'quantile': out_dim = 2
    else: raise NotImplementedError(f"loss function {config['loss_function_type']} has not been implemented")
    config['out_dim'] = out_dim
    # ...

    # Vision Model:
    '''None cause need to be set. Might be defined after if we use NetMob Data as input'''
    #'''
    #if 'vision_input_type' == 'image_per_stations': the model will extract feature by looking around a station
    #if 'vision_input_type' == 'unique_image_through_lyon':  the model will extract feature by looking through the entiere map, and then return N outputs (one for each station)
    #'''
    config['vision_model_name'] =  None  # -> Define the type of model used to extract contextual information from NetMob
    config['vision_input_type'] =  None  # 'image_per_stations' # 'unique_image_through_lyon'  

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
    module_path = f"dl_models.{args.model_name}.load_config"
    module = importlib.import_module(module_path)
    importlib.reload(module)
    locals()[f"args_{args.model_name}"] = module.args
    locals()[f"args_HP_{args.model_name}"] = module.args_HP 

    # Merge Args: 
    args = Namespace(**{**vars(args),**vars(locals()[f"args_{args.model_name}"]),**vars(locals()[f"args_HP_{args.model_name}"])})

    
    return(args)

def convert_into_parameters(config):
    parser = argparse.ArgumentParser()

    for key in config.keys():
        default = config[key]
        parser.add_argument(f'--{key}', type=type(default), default=default)

    args = parser.parse_args(args=[])
    return(args)

def update_out_dim(args):
    if args.loss_function_type == 'MSE': 
        args.out_dim = 1
        args.alpha = None
    elif args.loss_function_type == 'quantile': 
        args.out_dim = 2
    else: 
        raise NotImplementedError(f'loss function {args.loss_function_type} has not been implemented')
    return args 

def update_modif(args,name_gpu='cuda'):
    #Update modification:
    args = update_out_dim(args)
    #...
    
    # Ray tuning function can't hundle with PyTorch multiprocessing : 
    if args.ray:
        args.num_workers = 0
        args.persistent_workers = False
        args.track_pi = False
    # ...

    # Modif about n_vertex: 
    if args.set_spatial_units is not None:
        args.n_vertex = len(args.set_spatial_units)
    
    print(f">>>>Model: {args.model_name}; K_fold = {args.K_fold}; Loss function: {args.loss_function_type} ") 
    print(">>>> Prediction sur une UNIQUE STATION et non pas les 40 ") if args.single_station else None
    return(args)