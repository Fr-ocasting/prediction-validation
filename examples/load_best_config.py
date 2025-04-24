import sys
import os
import pandas as pd
import pickle
from argparse import Namespace
import torch

current_path = os.path.dirname(__file__)
working_dir = os.path.abspath(os.path.join(current_path, '..'))
if working_dir not in sys.path:
    sys.path.insert(0, working_dir)


from constants.paths import SAVE_DIRECTORY
from examples.train_and_visu_non_recurrent import get_ds
from constants.paths import SAVE_DIRECTORY
from high_level_DL_method import load_optimizer_and_scheduler
from dl_models.full_model import full_model
from trainer import Trainer

def load_args_of_a_specific_trial(trial_id,add_name_id,save_folder,fold_name):
    path_to_model_args = f"{working_dir}/{SAVE_DIRECTORY}/{save_folder}/best_models"
    dic_args = pickle.load(open(f"{path_to_model_args}/model_args.pkl",'rb'))
    args_models = dic_args['model'][f"{trial_id}{add_name_id}_f{fold_name}"]['args']
    args_models = Namespace(**args_models)
    args_models.ray = False
    return args_models


def load_best_config(trial_id = 'subway_in_STGCN_MSELoss_2024_08_21_14_50_2810',folder = 'save/HyperparameterTuning',metric = '_metric/Loss_model'):
    # Load HP-tuning results :
    df_hp_tuning =pd.read_csv(f'{working_dir}/{folder}/{trial_id}.csv')
    model_args = pickle.load(open(f'{working_dir}/{folder}/model_args.pkl','rb'))

    # Load common args through all tuning trials:
    args = model_args['model'][trial_id]['args']

    # Get best config :
    best_model = df_hp_tuning.sort_values(metric).iloc[0]

    # Set tuned parameter from best config to 'args':
    HP_args = [indx.replace('config/', '') for indx in best_model.index if 'config/' in indx]
    
    print('\n>>>> Load best CONFIG')
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

    # Load covergae : 
    return(args)


def get_trainer_and_ds_from_saved_trial(trial_id,add_name_id,save_folder,modification,fold_to_evaluate = None):
    # Load Data and Init Model:
    if fold_to_evaluate is None:
        fold_name = 'complete_dataset'
    else:
        fold_name = fold_to_evaluate[0]

    #args,_ = load_configuration(trial_id1,load_config=True)
    args = load_args_of_a_specific_trial(trial_id,add_name_id,save_folder,fold_name)

    if fold_to_evaluate is None:  fold_to_evaluate = [args.K_fold-1]

    
           
    ds,_,_,_,_ =  get_ds(args_init=args,modification = modification,fold_to_evaluate=fold_to_evaluate)
    model = full_model(ds, args).to(args.device)


    # Load Trained Weights 
    model_param = torch.load(f"{working_dir}/{SAVE_DIRECTORY}/{save_folder}/best_models/{trial_id}{add_name_id}_f{fold_name}.pkl")
    model.load_state_dict(model_param['state_dict'],strict=True)


    # Load Trainer : 
    optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
    trainer = Trainer(ds,model,args,optimizer,loss_function,scheduler = scheduler)

    return trainer,ds,args
