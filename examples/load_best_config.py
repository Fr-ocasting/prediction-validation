import sys
import os

current_path = os.path.dirname(__file__)
working_dir = os.path.abspath(os.path.join(current_path, '..'))
if working_dir not in sys.path:
    sys.path.insert(0, working_dir)

import pandas as pd
import pickle
from argparse import Namespace
from constants.paths import SAVE_DIRECTORY

from utils.utilities_DL import get_loss,load_model_and_optimizer
from build_inputs.load_datasets_to_predict import load_datasets_to_predict
from calendar_class import get_time_slots_labels

def load_args_of_a_specific_trial(trial_id,add_name_id,save_folder,fold_name):
    dic_args = pickle.load(open(f"{working_dir}/{SAVE_DIRECTORY}/{save_folder}/best_models/model_args.pkl",'rb'))
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

if __name__ == '__main__':
    args = load_best_config(trial_id = 'subway_in_STGCN_MSELoss_2024_08_21_14_50_2810',folder = 'save/HyperparameterTuning',metric = '_metric/Loss_model')
    # Load model with the best config:
    dataset,_,_,_ = load_datasets_to_predict(args,coverage_period)
    _,dic_class2rpz,_,_ = get_time_slots_labels(dataset,nb_class = [0,1,2,3])
    loss_function = get_loss(args)
    model,optimizer,scheduler = load_model_and_optimizer(args,dic_class2rpz)

    print(model)