import sys
import os
import pandas as pd
import pickle
from argparse import Namespace
import torch
import ast 
import numpy as np 
current_path = os.path.dirname(__file__)
working_dir = os.path.abspath(os.path.join(current_path, '..'))
if working_dir not in sys.path:
    sys.path.insert(0, working_dir)

from constants.paths import SAVE_DIRECTORY
from pipeline.DataSet.load_datasets import get_ds
from constants.paths import SAVE_DIRECTORY
from pipeline.high_level_DL_method import load_optimizer_and_scheduler
from pipeline.Flex_MDI.Flex_MDI import full_model
from pipeline.trainer.trainer import Trainer


def load_best_config_from_HPO(trial_id = 'subway_in_STGCN_MSELoss_2024_08_21_14_50_2810',
                              folder = 'save/HyperparameterTuning',
                              metric = '_metric/Loss_model'
                              ):
    ''' Load the best configuration saved from the HPO module.
    Args:
    ----
    trial_id: str, the trial id of the HPO, which return the best configuration associated
    examples: 
        >>> trial_id = 'subway_in_STGCN_MSELoss_2024_08_21_14_50_2810'
    '''
        
    # Load HP-tuning results :
    df_hp_tuning =pd.read_csv(f'{working_dir}/{folder}/{trial_id}.csv')
    model_args = pickle.load(open(f'{working_dir}/{folder}/model_args.pkl','rb'))

    # Load common args through all tuning trials:
    args = model_args['model'][trial_id]['args']

    # Get best config :
    best_model = df_hp_tuning.sort_values(metric).iloc[0] 
    best_model = parse_lists_in_series(best_model)   # Avoid Ray saving  list of float/int as str. 

    # Set tuned parameter from best config to 'args':
    HP_args = [indx.replace('config/', '') for indx in best_model.index if 'config/' in indx]
    
    print('\n----- Load best CONFIG')
    for arg in HP_args:
        if 'vision_' in arg:
            arg_vision = arg.replace('vision_', '')
            if 'concatenation_order/' in arg_vision:
                arg_vision = arg_vision.replace('concatenation_order/','')
            if 'n_head_d_model/' in arg_vision:
                arg_vision = arg_vision.replace('n_head_d_model/','')   

            if 'grn_out_dim' in arg_vision:
                setattr(args['args_vision'],'out_dim',best_model[f'config/{arg}'])
                
            setattr(args['args_vision'],arg_vision,best_model[f'config/{arg}'])

        elif 'scheduler/' in arg:
            schedule_args = arg.replace('scheduler/', '')
            args[schedule_args] = best_model[f'config/{arg}']

        elif 'TE_' in arg:
            args_TE = arg.replace('TE_', '')
            if 'concatenation_order/' in args_TE:
                args_TE = args_TE.replace('concatenation_order/','')
            setattr(args['args_embedding'],args_TE,best_model[f'config/{arg}'])

        else :
            args[arg] = best_model[f'config/{arg}']

    # Transform 'dict' to 'Namespace' object: 
    args = Namespace(**args)

    # Update 
    args.ray = False  # !!!

    if not(hasattr(args,'hp_tuning_on_first_fold')):
        args.hp_tuning_on_first_fold = False

    folds = list(np.arange(args.K_fold))

    return args,folds

def parse_lists_in_series(pd_serie: pd.Series):
    """
    Function to :
    - parse str that supposed to be list of number in a pandas series.
    - convert np.int32 or np.int64 into native 'int'


    """
    for k, v in pd_serie.items():
        if isinstance(v, str):
            try:
                val = ast.literal_eval(v)
                if isinstance(val, list) and all(isinstance(x, (int, float)) for x in val):
                    pd_serie[k] = [int(x) if isinstance(x, np.integer) else x for x in val] # convert into natif 'int' (not int64)
                else:
                    pd_serie[k] = v
            except (ValueError, SyntaxError):
                pd_serie[k] = v
        else:
            if isinstance(v, list) and all(isinstance(x,(np.int32,np.int64)) for x in v):
                v = [int(x) for x in v]
            if isinstance(v, (np.int32, np.int64)):
                v = int(v)
            pd_serie[k] = v
    return pd_serie


def load_args_of_a_specific_trial(trial_id,add_name_id,save_folder,fold_name):
    path_to_model_args = f"{SAVE_DIRECTORY}/{save_folder}/best_models"
    dic_args = pickle.load(open(f"{path_to_model_args}/model_args.pkl",'rb'))
    args_models = dic_args['model'][f"{trial_id}{add_name_id}_f{fold_name}"]['args']
    args_models = Namespace(**args_models)
    args_models.ray = False
    return args_models



def load_trainer_ds_from_saved_trial(args=None,
                                     model_save_path=None,
                                     modification={},
                                     ds_init = None,
                                     args_init= None,
                                     trial_id = None,
                                     add_name_id = '',
                                     save_folder = None,
                                     fold_to_evaluate = None
                                     ):
    """
    
    Examples:
    Either args_init is None: 
    >>> args_init = Namespace(**args)

    Either args_init is not None:

    """

    # Choice 1: args_init & ds_init is None: ---
    if ds_init is None: 
        # ----
        if trial_id is not None:
                # Load Data and Init Model:
                if fold_to_evaluate is None:
                    fold_name = 'complete_dataset'
                    fold_to_evaluate = [args.K_fold-1]
                else:
                    fold_name = fold_to_evaluate[0]

                if args is None: 
                    args_init = load_args_of_a_specific_trial(trial_id,add_name_id,save_folder,fold_name)
                else:
                    if type(args) == dict:
                        args_init = Namespace(**args)
                    else:
                        args_init = Namespace(**vars(args))

                if save_folder is not None:
                    model_save_path = f"{SAVE_DIRECTORY}/{save_folder}/best_models/{trial_id}{add_name_id}_f{fold_name}.pkl"
                else: 
                    raise NotImplementedError("Either 'save_folder' or 'model_save_path' must be provided to load the model parameters.")
        # ----
        else:
            if type(args) == dict:
                args_init = Namespace(**args)
            else:
                args_init = Namespace(**vars(args))

        fold_to_evaluate = [args.K_fold-1] if fold_to_evaluate is None else fold_to_evaluate
        args_init.ray = False

        ds,args_updated,_,_,_ =  get_ds(args_init=args_init,modification = modification,fold_to_evaluate=fold_to_evaluate)
    # -----

    # Choice 3: args_init & ds_init are defined : ---
    elif (ds_init is not None) and (args_init is not None):
        ds = ds_init
        args_updated = args_init
        for key,values in modification.items():
            setattr(args_updated,key,values)
    # -----
   
    else:
        raise NotImplementedError("Either 'ds_init' and 'args_init' or 'trial_id' must be provided to load the model parameters.")


    # Load Model: 
    model = full_model(ds, 
                       args_updated
                       ).to(args_updated.device)


    model_param = torch.load(f"{model_save_path}")

    # --- Load state dict : 
    prefix = "_orig_mod."
    model_param['state_dict'] = {k[len(prefix):] if k.startswith(prefix) else k: v    #Needed here to remove the prefix added by torch.compile
                for k, v in model_param['state_dict'].items()}
    
    model.load_state_dict(model_param['state_dict'],strict=True)
    model = model.to(args_updated.device)
    # ---
    
    optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args_updated)
    trainer = Trainer(ds,model,args_updated,optimizer,loss_function,scheduler = scheduler)

    return trainer, ds, args_updated