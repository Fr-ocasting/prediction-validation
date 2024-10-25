import sys
import os

# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np 
from train_model_on_k_fold_validation import train_valid_K_models,save_model_metrics
from utils.utilities_DL import match_period_coverage_with_netmob
from constants.config import get_args,update_modif,update_args
from constants.paths import file_name,folder_path,SAVE_DIRECTORY
from utils.save_results import get_date_id
from K_fold_validation.K_fold_validation import KFoldSplitter
from high_level_DL_method import load_model,load_optimizer_and_scheduler
from trainer import Trainer


def local_get_args(model_name):
    # Load base args
    args = get_args(model_name)

    # Modification :
    args.epochs = 1 
    args.W = 0
    args.K_fold = 6   # Means we will use the first fold for the Ray Tuning and the 4 other ones to get the metrics
    args.ray = False
    args.loss_function_type = 'MSE'  #'MSE' # 'quantile'
    args.scheduler = None

    #  evaluation on the first fold only :
    hp_tuning_on_first_fold = True # True # False // if True, then we remove the first fold as we consid we used it for HP-tuning
    args.evaluate_complete_ds = True  # True # False // if True, then evaluation also on the entiere ds 

    # update each modif
    args = update_modif(args)
    
    # set number of folds to evaluate
    folds =  [0]

    # set total coverage period 
    coverage = match_period_coverage_with_netmob(file_name)

    return(args,folds,coverage,hp_tuning_on_first_fold)

def get_trial_id(args,dataset_names,vision_model_name=None):
    date_id = get_date_id()
    datasets_names = '_'.join(dataset_names)
    model_names = '_'.join([args.model_name,vision_model_name]) if 'netmob' in dataset_names  else args.model_name
    trial_id =  f"{datasets_names}_{model_names}_{args.loss_function_type}Loss_{date_id}"
    return trial_id

def get_inputs(dataset_names,args,coverage,folder_path,file_name,vision_model_name,folds):
    K_fold_splitter = KFoldSplitter(dataset_names,args,coverage,folder_path,file_name,vision_model_name,folds)
    K_subway_ds,dic_class2rpz,_ = K_fold_splitter.split_k_fold()
    return(K_fold_splitter,K_subway_ds,dic_class2rpz)


def train_on_ds(model_name,ds,args,trial_id,save_folder,dic_class2rpz,df_loss):
    model = load_model(args,dic_class2rpz)
    optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
    trainer = Trainer(ds,model,args,optimizer,loss_function,scheduler = scheduler,dic_class2rpz = dic_class2rpz,show_figure = False,trial_id = trial_id, fold=0,save_folder = save_folder)
    trainer.train_and_valid(mod = 1000,mod_plot = None) 

    df_loss[f"{model_name}_train_loss"] = trainer.train_loss
    df_loss[f"{model_name}_valid_loss"] = trainer.valid_loss

    return(trainer,df_loss)

if __name__ == '__main__':

    # GET PARAMETERS
    dataset_names = ["subway_in"] # ["subway_in","calendar"] # ["subway_in"]
    vision_model_name = None
    save_folder = 'benchmark/fold0/'
    df_loss = pd.DataFrame()

    model_name ='STGCN' #'MTGNN' # 'STGCN'  #'CNN' # 'DCRNN'
    (args,folds,coverage,hp_tuning_on_first_fold) = local_get_args(model_name)
    print(f"\nModel perf on {dataset_names} with {model_name}")

    trial_id = get_trial_id(args,dataset_names,vision_model_name=None)
    K_fold_splitter,K_subway_ds,dic_class2rpz = get_inputs(dataset_names,args,coverage,folder_path,file_name,vision_model_name,folds)
    ds = K_subway_ds[0]

    trainer,df_loss = train_on_ds(model_name,ds,args,trial_id,save_folder,dic_class2rpz,df_loss)

    for model_name in ['MTGNN','CNN','DCRNN']:
        (args,folds,coverage,hp_tuning_on_first_fold) = local_get_args(model_name)
        args = update_args(args,ds,dataset_names)
        print(f"\nModel perf on {dataset_names} with {model_name}")
        trial_id = get_trial_id(args,dataset_names,vision_model_name=None)

        trainer,df_loss = train_on_ds(model_name,ds,args,trial_id,save_folder,dic_class2rpz,df_loss)
