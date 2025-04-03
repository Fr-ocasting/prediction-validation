import re 
import numpy as np 
import pandas as pd
from argparse import Namespace
import pickle 
import torch 

import sys
import os

current_path = notebook_dir = os.getcwd()
working_dir = os.path.abspath(os.path.join(current_path, '..','..'))
if working_dir not in sys.path:
    sys.path.insert(0, working_dir)
    
from high_level_DL_method import load_model,load_optimizer_and_scheduler
from trainer import Trainer
from examples.train_and_visu_non_recurrent import get_multi_ds
from utils.metrics import evaluate_metrics

SAVE_FOLDER_PATH = f'{current_path}/save/K_fold_validation/training_with_HP_tuning'

import re 

L_Apps = ['Apple_Video','Google_Play_Store','Google_Maps','Web_Clothes','Uber', 'Twitter',
        'Microsoft_Mail', 'Microsoft_Store', 'Apple_Music', 'Microsoft_Office', 'Pokemon_GO', 'Clash_of_Clans', 'Yahoo_Mail', 'PlayStation',
        'Wikipedia', 'Apple_Web_Services', 'Pinterest', 'Web_Ads', 'Google_Mail', 'Google_Meet',
        'Apple_Siri', 'Web_Adult', 'Spotify', 'Deezer', 'Waze', 'Web_Games', 'Apple_App_Store', 'Microsoft_Skydrive', 'Google_Docs', 'Microsoft_Web_Services',
        'Molotov', 'YouTube', 'Apple_iTunes', 'Apple_iMessage', 'DailyMotion', 'Netflix', 'Web_Transportation',
        'Web_Downloads', 'SoundCloud', 'TeamViewer', 'Google_Web_Services', 'Facebook', 'EA_Games', 'Tor', 'Amazon_Web_Services',
        'Web_e-Commerce', 'Telegram', 'Apple_Mail','Dropbox', 'Web_Food', 'Apple_iCloud', 'Skype', 'Facebook_Messenger', 'Twitch', 'Microsoft_Azure',
        'Instagram', 'Facebook_Live', 'Web_Streaming', 'Orange_TV', 'Periscope', 'Snapchat' ,'Web_Finance' ,'WhatsApp', 'Web_Weather','Google_Drive','LinkedIn','Yahoo','Fortnite']


def get_df_results(trial_id,model_args,L_Apps,split_key = 'eps100_'):
    df = pd.DataFrame(columns = ['mse','mae','mape','fold','id','trial_num'])
    for app in L_Apps:
        pattern = re.compile(rf"{trial_id}_{app}(_\d+)?_f")
        best_model_names = [name for name in model_args['model'].keys() if pattern.search(name)]

        for trial_fold in best_model_names:
            model_metrics = model_args['model'][trial_fold]['performance']['test_metrics']
            pattern_1 = re.compile(rf"{trial_id}_{app}_f")
            pattern_2 = re.compile(rf"{trial_id}_{app}_\d_f")
            if pattern_1.search(trial_fold):
                pattern_1_i = re.compile(rf"{trial_id}_{app}_f\d$")
                trial_num = 1
                k = trial_fold.split('f')[-1] if pattern_1_i.search(trial_fold) else model_args['model'][trial_fold]['args']['K_fold']-1-model_args['model'][trial_fold]['args']['hp_tuning_on_first_fold']
            elif pattern_2.search(trial_fold):
                pattern_2_i = re.compile(rf"{trial_id}_{app}_\d_f\d$")
                trial_num = trial_fold.split('f')[-2].split('_')[-2]
                k = trial_fold.split('f')[-1] if pattern_2_i.search(trial_fold) else model_args['model'][trial_fold]['args']['K_fold']-1-model_args['model'][trial_fold]['args']['hp_tuning_on_first_fold']
            else:
                raise NotImplementedError
            
            df.loc[len(df)] = [model_metrics['mse'],model_metrics['mae'],model_metrics['mape'],k,app,trial_num]
    return df


def load_trained_model(model_args,ds,model_fold_i,folder_name):
    if folder_name != '':
        save_path = f'{SAVE_FOLDER_PATH}/{folder_name}/best_models'
    else:
        save_path = f'{SAVE_FOLDER_PATH}/best_models'
    selected_model_path = f"{save_path}/{model_fold_i}.pkl"
    
    model_param = torch.load(selected_model_path)
    args = model_args['model'][model_fold_i]['args']
    args = Namespace(**args)
    model = load_model(ds, args)
    model.load_state_dict(model_param['state_dict'])
    optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
    trainer = Trainer(ds,model,args,optimizer,loss_function,scheduler = scheduler)
    return trainer 

def get_metrics_from_test(trainer,ds,metric_list = ['mse','mae','mape','mase']):
    Preds,Y_true,T_labels = trainer.testing(ds.normalizer)
    dic_pred_metrics = evaluate_metrics(Preds,Y_true,metric_list)
    return dic_pred_metrics

def load_model_args(folder_name):
    '''Return the dictionnary of all the saved model. 
    dictionaries are saved in a sub-folder, that's why we need to set a 'folder_name'.

    If we want to load trained model which comes directly from the HP-tuning:
    >>> set folder_name = ''
    
     '''
    if folder_name != '':
        save_path = f'{SAVE_FOLDER_PATH}/{folder_name}/best_models'
    else:
        save_path = f'{SAVE_FOLDER_PATH}/best_models'
    model_args = pickle.load(open(f'{save_path}/model_args.pkl','rb'))   
    return model_args

def load_K_fold_datasets(model_args,model_id_i):
    """
    Load K-fold dataset associated to a saved model. A saved model is trained on a speicfic fold. 
    Whatch-out The associated metrics to the model would be different according to which fold we consider.
    """

    args_0 = model_args['model'][model_id_i]['args']
    args_0 = Namespace(**args_0)
    args_with_contextual,K_subway_ds = get_multi_ds(args_0.model_name, args_0.dataset_names,args_0.dataset_for_coverage,args_init = args_0,fold_to_evaluate = np.arange(args_0.K_fold))
    return K_subway_ds

def get_masked_previous_preds_true(mask,Preds,Y_true):
    """
    args:
    -----
    mask : has to be torch.Tensor of bool (True/False)
    """
    mask_previous = mask[1:]
    previous = Y_true[:-1]
    masked_previous = previous[mask_previous]
    masked_Preds = Preds[1:][mask_previous]
    masked_True = Y_true[1:][mask_previous]
    return masked_previous,masked_Preds,masked_True

if __name__ == '__main__':
    folder_name = 'tmps'
    save_path = f'save/K_fold_validation/training_with_HP_tuning/{folder_name}/best_models'
    model_args = pickle.load(open(f'{current_path}/{save_path}/model_args.pkl','rb'))

    trial_id = 'subway_in_subway_out_STGCN_MSELoss_2025_02_19_00_05_19271'

    model_i = f"{trial_id}_1_f0"
    args_0 = model_args['model'][model_i]['args']
    args_0 = Namespace(**args_0)
    args_with_contextual,K_subway_ds = get_multi_ds(args_0.model_name, args_0.dataset_names,args_0.dataset_for_coverage,args_init = args_0,fold_to_evaluate = np.arange(args_0.K_fold))

    k_fold = 0
    ds = K_subway_ds[k_fold]
    selected_model_path = f"{current_path}/{save_path}/{model_i}.pkl"
    trainer = load_trained_model(model_args,ds,model_i,folder_name)
    Preds,Y_true,T_labels = trainer.testing(ds.normalizer)