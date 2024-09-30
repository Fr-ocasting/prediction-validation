import sys
import os

# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np 
from trainer import Trainer
from K_fold_validation.K_fold_validation import KFoldSplitter
from constants.paths import folder_path,file_name,SAVE_DIRECTORY
from trainer import Trainer
from high_level_DL_method import load_model,load_optimizer_and_scheduler
import numpy as np 


def train_model_on_k_fold_validation(trial_id,load_config,save_folder,epochs=None,folder = 'save/HyperparameterTuning'):
    '''
    1. Load the best config according to our HP-Tuning
    2. Apply the K-fold validation to split inputs
    3. For each fold, load a new model and train it with the associated fold of inputs
    4. Keep track on train/valid losses and the best results. Save them.
    '''
    # If Load config: 
    if load_config:
        from examples.load_best_config import load_best_config
        args,coverage = load_best_config(trial_id)
        #Update modif validation
        args.train_prop = 0.6
        args.valid_prop = 0.2
        args.test_prop = 0.2
        
        #Change/Set epochs: 
        if epochs is not None:
            args.epochs = epochs
        dataset_names = args.dataset_names 
        vision_model_name = args.args_vision.model_name if len(vars(args.args_vision))>0 else None

    # If new config : 
    else:
        from examples.load_random_config import args,coverage,trial_id,dataset_names,vision_model_name


    # Sliding Window Cross Validation 
    ## Define fixed Dataset K_fold split for each trial: 
    folds = list(np.arange(args.K_fold))
    K_fold_splitter = KFoldSplitter(dataset_names,args,coverage,folder_path,file_name,vision_model_name,folds)
    K_subway_ds,dic_class2rpz,_ = K_fold_splitter.split_k_fold()

    ''' Plotting if necessary: '''
    #from plotting.plotting import plot_k_fold_split
    #plot_k_fold_split(K_subway_ds,K_subway_ds[0].init_invalid_dates)
    ''' ______________________ ''' 

    ## Split Tuning and Validation datasets:
    # ds_tuning = K_subway_ds[0]
    ds_validation = K_subway_ds[1:]
    del K_subway_ds

    ## Train on the K-1 folds:

    #___ Init
    valid_losses = []
    for training_mode in ['valid','test']:
        for metric in ['mse','mae','mape']:
            globals()[f'{training_mode}_{metric}'] = []

    training_mode_list = ['valid','test']
    metric_list = ['mse','mae','mape']
    df_loss = pd.DataFrame()

    #___ Through each fold : 
    for fold,ds in enumerate(ds_validation):
        model = load_model(args,dic_class2rpz)
        optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
        trainer = Trainer(ds,model,args,optimizer,loss_function,scheduler = scheduler,dic_class2rpz = dic_class2rpz,show_figure = False,trial_id = trial_id, fold=fold,save_folder = save_folder)
        trainer.train_and_valid(mod = 1000,mod_plot = None) 

        df_loss[f"f{fold}_train_loss"] = trainer.train_loss
        df_loss[f"f{fold}_valid_loss"] = trainer.valid_loss

        valid_losses.append(trainer.performance['valid_loss'])

        # Keep track on metrics :
        for training_mode in training_mode_list:
            for metric in metric_list:          
                l = trainer.performance[f'{training_mode}_metrics'][metric]   
                globals()[f'{training_mode}_{metric}'].append(l)

    ## Save Model: 
    row = {f"fold{k}": [loss] for k,loss in enumerate(valid_losses)}
    row.update({'mean' : [np.mean(valid_losses)]})
    df_results = pd.DataFrame.from_dict(row)
    df_results.to_csv(f"{SAVE_DIRECTORY}/{save_folder}/VALID_{trial_id}.csv")


    df_loss.to_csv(f"{SAVE_DIRECTORY}/{save_folder}/Losses_{trial_id}.csv")
    
    
    df_metrics = pd.DataFrame(index = training_mode_list, 
                            data = {metric : [np.mean(globals()[f'{training_mode}_{metric}']) for training_mode in training_mode_list] for metric in metric_list}
                            )
    df_metrics.to_csv(f"{SAVE_DIRECTORY}/{save_folder}/METRICS_{trial_id}.csv")



# ========================================================
# Application 
# ========================================================
if __name__ == '__main__':
    #'subway_in_STGCN_MSELoss_2024_08_25_18_05_25229'
    #'subway_in_calendar_STGCN_MSELoss_2024_08_25_22_56_92429'
    #'netmob_subway_in_STGCN_ImageAvgPooling_MSELoss_2024_08_24_01_42_17375'
    #'netmob_subway_in_STGCN_FeatureExtractor_ResNetInspired_MSELoss_2024_08_23_06_53_46982'
    #'netmob_subway_in_calendar_STGCN_ImageAvgPooling_MSELoss_2024_08_27_00_16_90667'
    #'netmob_subway_in_calendar_STGCN_FeatureExtractor_ResNetInspired_MSELoss_2024_08_28_06_04_41108'
    #'subway_in_STGCN_MSELoss_2024_08_21_14_50_2810'

    load_config = True
    save_folder = 'K_fold_validation'
    trial_id = 'subway_in_STGCN_MSELoss_2024_08_25_18_05_25229'
    epochs = 500


    train_model_on_k_fold_validation(trial_id,load_config,save_folder,epochs,folder = 'save/HyperparameterTuning')