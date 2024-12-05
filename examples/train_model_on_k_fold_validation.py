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
from constants.paths import SAVE_DIRECTORY
from trainer import Trainer
from high_level_DL_method import load_model,load_optimizer_and_scheduler
import numpy as np 

def load_configuration(trial_id,load_config,epochs):
    # If Load config: 
    if load_config:
        from examples.load_best_config import load_best_config
        args = load_best_config(trial_id)
        #Update modif validation
        args.train_prop = 0.6
        args.valid_prop = 0.2
        args.test_prop = 0.2
        
        #Change/Set epochs: 
        if epochs is not None:
            args.epochs = epochs
        dataset_names = args.dataset_names 
        vision_model_name = args.args_vision.model_name if len(vars(args.args_vision))>0 else None
        folds = list(np.arange(args.K_fold))

    # If new config : 
    else:
        from examples.load_random_config import args,dataset_names,vision_model_name,folds

    return args,dataset_names,vision_model_name,folds

def train_valid_K_models(args,vision_model_name,folds,hp_tuning_on_first_fold,trial_id,save_folder):
    '''
    args:
    ------
    folds: list of folds (within [0,args.K-fold]) we would like to evaluate
    hp_tuning_on_first_fold : if True, then we remove the fold 0, which has been used for HP-tuning
    '''
        
    # Return a list of K-fold Dataset:
    K_fold_splitter = KFoldSplitter(args,vision_model_name,folds)
    K_subway_ds,dic_class2rpz,_ = K_fold_splitter.split_k_fold()

    # Keep the first fold or not : 
    if hp_tuning_on_first_fold :
        ds_validation = K_subway_ds[1:]
    else:
        ds_validation = K_subway_ds
    del K_subway_ds

    ## Specific case if we want to validate on the init entiere dataset:
    if args.evaluate_complete_ds: 
        args.train_prop = 0.6
        args.valid_prop = 0.2
        args.test_prop = 0.2
        subway_ds,_,_,dic_class2rpz = K_fold_splitter.load_init_ds(normalize = True)
        ds_validation.append(subway_ds)
        del subway_ds

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
        # ____ Specific case if we want to validate on the init entiere dataset:
        condition = (args.evaluate_complete_ds) and (fold == len(ds_validation)-1)
        if condition:
            fold = 'complete_dataset'

        model = load_model(ds, args,dic_class2rpz)
        optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
        trainer = Trainer(ds,model,args,optimizer,loss_function,scheduler = scheduler,dic_class2rpz = dic_class2rpz,show_figure = False,trial_id = trial_id, fold=fold,save_folder = save_folder)
        trainer.train_and_valid(mod = 1000,mod_plot = None) 

        df_loss[f"f{fold}_train_loss"] = trainer.train_loss
        df_loss[f"f{fold}_valid_loss"] = trainer.valid_loss


        # ____ Only keep metrics from k-folds (and not about the training on the entiere dataset):
        if not(condition):
            valid_losses.append(trainer.performance['valid_loss'])

        # Keep track on metrics :
        for training_mode in training_mode_list:
            for metric in metric_list:          
                l = trainer.performance[f'{training_mode}_metrics'][metric]   
                globals()[f'{training_mode}_{metric}'].append(l)
    
    return trainer,args,valid_losses,training_mode_list,metric_list,df_loss


def get_model_metrics(trainer,args,valid_losses,training_mode_list,metric_list):
    row = {f"fold{k}": [loss] for k,loss in enumerate(valid_losses)}
    row.update({'mean' : [np.mean(valid_losses)]})
    if (args.evaluate_complete_ds):
        row.update({'complete_dataset': trainer.performance['valid_loss']})  # The associated validation is from the last trained model
    df_results = pd.DataFrame.from_dict(row)

    dict_data_metric = {metric : [np.mean(globals()[f'{training_mode}_{metric}']) for training_mode in training_mode_list] for metric in metric_list}
    if (args.evaluate_complete_ds):
        dict_data_metric.update({f'{metric}_complete_ds':[trainer.performance[f'{training_mode}_metrics'][metric] for training_mode in training_mode_list ] for metric in metric_list})
    df_metrics = pd.DataFrame(index = training_mode_list, 
                            data = dict_data_metric
                            )   
    
    return df_results,df_metrics
    
def save_model_metrics(trainer,args,valid_losses,training_mode_list,metric_list,df_loss,save_folder,trial_id):
    df_results,df_metrics =  get_model_metrics(trainer,args,valid_losses,training_mode_list,metric_list)

    df_results.to_csv(f"{SAVE_DIRECTORY}/{save_folder}/VALID_{trial_id}.csv")
    df_loss.to_csv(f"{SAVE_DIRECTORY}/{save_folder}/Losses_{trial_id}.csv")
    df_metrics.to_csv(f"{SAVE_DIRECTORY}/{save_folder}/METRICS_{trial_id}.csv")


def train_model_on_k_fold_validation(trial_id,load_config,save_folder,epochs=None,hp_tuning_on_first_fold= True):
    '''
    1. Load the best config according to our HP-Tuning
    2. Apply the K-fold validation to split inputs
    3. For each fold, load a new model and train it with the associated fold of inputs
    4. Keep track on train/valid losses and the best results. 
    5. Save them.
    '''
    # 1. Load the best config according to our HP-Tuning / Or Load random config :
    args,dataset_names,vision_model_name,folds = load_configuration(trial_id,load_config,epochs)

    # 2. 3. 4. 
    trainer,args,valid_losses,training_mode_list,metric_list,df_loss = train_valid_K_models(args,vision_model_name,folds,hp_tuning_on_first_fold,trial_id,save_folder)
    # 5.
    save_model_metrics(trainer,args,valid_losses,training_mode_list,metric_list,df_loss,save_folder,trial_id)


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




    # Case 1. HP tuning have been computed on the first fold. We are training on the K-1 other folds
    # ----------
    if False:
        save_folder = 'K_fold_validation/training_with_HP_tuning'
        load_config = True
        trial_id = 'subway_in_STGCN_MSELoss_2024_08_25_18_05_25229'
        epochs = 500
        train_model_on_k_fold_validation(trial_id,load_config,save_folder,epochs,hp_tuning_on_first_fold = True)


    # Case 2. We just need to test some configuration, where we set the configuration from 'load_random_config.py:
    # ----------
    if True: 
        save_folder = 'K_fold_validation/traing_without_HP_tuning'
        if not os.path.exists(f"{SAVE_DIRECTORY}/{save_folder}"):
            os.mkdir(f"{SAVE_DIRECTORY}/{save_folder}")

        load_config = False
        trial_id = 'train_random_config'
        epochs = 200
        train_model_on_k_fold_validation(trial_id,load_config,save_folder,epochs,hp_tuning_on_first_fold=False)


