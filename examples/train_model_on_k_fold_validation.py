import sys
import os

# Get Parent folder : 
current_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

import pandas as pd
import numpy as np 
from pipeline.trainer import Trainer
from pipeline.K_fold_validation.K_fold_validation import KFoldSplitter
from constants.paths import SAVE_DIRECTORY
from pipeline.trainer import Trainer
from pipeline.high_level_DL_method import load_optimizer_and_scheduler
from pipeline.dl_models.full_model import full_model
from examples.train_and_visu_non_recurrent import get_multi_ds
import numpy as np 

def load_configuration(trial_id,load_config,modification = None,folder =  'save/HyperparameterTuning'):
    # If Load config: 
    if load_config:
        from examples.load_best_config import load_best_config
        args = load_best_config(trial_id, folder = folder)
        #Update modif validation
        #args.train_prop = 0.6
        #args.valid_prop = 0.2
        #args.test_prop = 0.2
        args.hp_tuning_on_first_fold = True
        folds = list(np.arange(args.K_fold))

    # If new config : 
    else:
        from examples.load_random_config import get_default_args
        args,folds = get_default_args(modification)
        
    return args,folds

def load_k_fold_dataset(args,folds):
    K_fold_splitter = KFoldSplitter(args, folds=folds)
    K_subway_ds,_ = K_fold_splitter.split_k_fold()

    # Keep the first fold or not : 
    if args.hp_tuning_on_first_fold :
        ds_validation = K_subway_ds[1:]
    else:
        ds_validation = K_subway_ds
    del K_subway_ds

    ## Specific case if we want to validate on the init entiere dataset:
    if (args.evaluate_complete_ds and args.validation_split_method == 'custom_blocked_cv'): 
        subway_ds,_,_ = K_fold_splitter.load_init_ds(normalize = True)
        ds_validation.append(subway_ds)
        del subway_ds
    return(ds_validation,args)

def init_metrics(args):
    #___ Init
    training_mode_list = ['valid','test']
    if args.loss_function_type == 'quantile':
        metric_list = ['MPIW','PICP']
    else:
        metric_list = args.metrics
    valid_losses = []
    df_loss = pd.DataFrame()

    dic_results = {}
    for training_mode in training_mode_list:
        for metric in metric_list:
            for h in range(args.horizon_step,args.step_ahead+1,args.horizon_step):  
                dic_results[f'{training_mode}_{metric}_h{h}'] = []
    # ...
    return valid_losses,df_loss,training_mode_list,metric_list,dic_results

def load_trainer(args,trial_id,save_folder=None,modification={},fold_to_evaluate = None):
    if fold_to_evaluate is None : 
        fold = args.K_fold-1
        fold_to_evaluate = [fold]
    else:
        if len(fold_to_evaluate) == 1:
            fold = fold_to_evaluate[0]
        else:
            raise NotImplementedError(f'fold_to_evaluate has to be None or a list of one single element. Here fold_to_evaluate= {fold_to_evaluate}')
        
    args,ds_validation = get_multi_ds(args.model_name,
                                    args.dataset_names,
                                    args.dataset_for_coverage,
                                    modification = modification,
                                    args_init = args, 
                                    fold_to_evaluate = fold_to_evaluate)
    
    ds = ds_validation[0]
    model = full_model(ds, args).to(args.device)
    optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
    trainer = Trainer(ds,model,args,optimizer,loss_function,scheduler = scheduler,show_figure = False,trial_id = trial_id, fold=fold,save_folder = save_folder)
    return trainer, args, ds

def train_valid_1_model(args,trial_id,save_folder,modification={},fold_to_evaluate=None):
    # Return a list of K-fold Dataset:
    if fold_to_evaluate is None : fold_to_evaluate = [args.K_fold-1]
    trainer, args, ds = load_trainer(args,trial_id,save_folder,modification,fold_to_evaluate)
    trainer.train_and_valid(normalizer = ds.normalizer,mod = 1000,mod_plot = None) 

    valid_losses,df_loss,training_mode_list,metric_list,dic_results= init_metrics(args)

    return trainer,args,training_mode_list,metric_list

def get_conditions(args,fold_i,ds_validation):
    condition1 = (args.evaluate_complete_ds) and (fold_i == len(ds_validation)-1)
    condition2 = condition1 and args.validation_split_method == 'forward_chaining_cv'
    if condition1:
        fold = 'complete_dataset'
    else:
        fold = fold_i
    return condition1,condition2,fold

def keep_track_on_metrics(trainer,args,df_loss,valid_losses,dic_results,fold_i,fold,condition1,condition2,training_mode_list,metric_list):
    if condition2: 
        df_loss[f"f{fold_i}_train_loss"] = trainer.train_loss
        df_loss[f"f{fold_i}_valid_loss"] = trainer.valid_loss

    df_loss[f"f{fold}_train_loss"] = trainer.train_loss
    df_loss[f"f{fold}_valid_loss"] = trainer.valid_loss

    # ____ Only keep metrics from k-folds (and not from the training on the added 'complete_dataset'):
    if not(condition1) or (condition2):
        valid_losses.append(trainer.performance['valid_loss'])

    # Keep track on metrics :
    for training_mode in training_mode_list:
        for metric in metric_list: 
            for h in range(args.horizon_step,args.step_ahead+1,args.horizon_step):       
                l = trainer.performance[f'{training_mode}_metrics'][f"{metric}_h{h}"]   
                dic_results[f'{training_mode}_{metric}_h{h}'].append(l)

    return df_loss, valid_losses,dic_results 

def train_valid_K_models(args,trial_id,save_folder,modification={}):
    '''
    args:
    ------
    folds: list of folds (within [0,args.K-fold]) we would like to evaluate
    hp_tuning_on_first_fold : if True, then we remove the fold 0, which has been used for HP-tuning
    '''
        
    # Return a list of K-fold Dataset:
    args,ds_validation = get_multi_ds(args.model_name,
                                    args.dataset_names,
                                    args.dataset_for_coverage,
                                    modification = modification,
                                    args_init = args, 
                                    fold_to_evaluate = np.arange(args.K_fold))
    

    ## Train on the K-1 folds:
    #___ Init

    valid_losses,df_loss,training_mode_list,metric_list,dic_results = init_metrics(args)
    # ...

    #___ Through each fold : 
    for fold_i,ds in enumerate(ds_validation):
        # ____ Specific case if we want to validate on the init entiere dataset:

        condition1,condition2,fold = get_conditions(args,fold_i,ds_validation)

        model = full_model(ds, args).to(args.device)
        optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
        trainer = Trainer(ds,model,args,optimizer,loss_function,scheduler = scheduler,show_figure = False,trial_id = trial_id, fold=fold,save_folder = save_folder)
        trainer.train_and_valid(normalizer = ds.normalizer,mod = 1000,mod_plot = None) 

        df_loss, valid_losses,dic_results = keep_track_on_metrics(trainer,args,df_loss,valid_losses,dic_results,fold_i,fold,condition1,condition2,training_mode_list,metric_list)
    
    return trainer,args,valid_losses,training_mode_list,metric_list,df_loss,dic_results


def get_model_metrics(trainer,args,valid_losses,training_mode_list,metric_list,dic_results):
    # Metrics Valid 
    row = {f"fold{k}": [loss] for k,loss in enumerate(valid_losses)}
    row.update({'mean' : [np.mean(valid_losses)]})
    if (args.evaluate_complete_ds):
        row.update({'complete_dataset': trainer.performance['valid_loss']})  # The associated validation is from the last trained model
    df_results = pd.DataFrame.from_dict(row)
    # ...

    # Metrics Test :
    model_metrics =  dic_results[f'{training_mode_list[0]}_{metric_list[0]}_h{args.horizon_step}']
    nb_folds = len(model_metrics)
    multi_cols =  pd.MultiIndex.from_product([metric_list, range(nb_folds),range(args.step_ahead)], 
                                            names=["metric", "fold","horizon"])

    df_metrics_by_folds = pd.DataFrame(index=training_mode_list, columns=multi_cols)

    for training_mode in training_mode_list:
        for metric in metric_list:
            for h in range(args.horizon_step,args.step_ahead+1,args.horizon_step):  
                model_metrics = dic_results[f'{training_mode}_{metric}_h{h}']
                for fold_idx, value in enumerate(model_metrics):
                    df_metrics_by_folds.loc[training_mode, (metric, fold_idx,h)] = value

    # Metrics K-folds: 
    dict_metrics_on_K_fold = {}

    # display metrics values in columns :
    if False:
        for metric in metric_list:
            print(f'\ntest_{metric}')
            for h in range(args.horizon_step,args.step_ahead+1,args.horizon_step):  
                print(dic_results[f'test_{metric}_h{h}'])


    mean_on_K_fold = {f"{metric}_h{h}" : [np.mean(dic_results[f'{training_mode}_{metric}_h{h}']) for training_mode in training_mode_list] for metric in metric_list for h in range(args.horizon_step,args.step_ahead+1,args.horizon_step)}
    var_on_K_fold = {f"VAR_{metric}_h{h}" : [np.var(dic_results[f'{training_mode}_{metric}_h{h}']) for training_mode in training_mode_list] for metric in metric_list for h in range(args.horizon_step,args.step_ahead+1,args.horizon_step)}
    dict_metrics_on_K_fold.update(mean_on_K_fold)
    dict_metrics_on_K_fold.update(var_on_K_fold)
    if (args.evaluate_complete_ds):
        dict_metrics_on_K_fold.update({f'{metric}_h{h}_complete_ds':[trainer.performance[f'{training_mode}_metrics'][f"{metric}_h{h}"] for training_mode in training_mode_list ] for metric in metric_list for h in range(args.horizon_step,args.step_ahead+1,args.horizon_step)})
    df_metrics = pd.DataFrame(index = training_mode_list, 
                            data = dict_metrics_on_K_fold
                            )   
    # ...
    
    return df_results,df_metrics,df_metrics_by_folds
def save_model_metrics(trainer,args,valid_losses,training_mode_list,metric_list,df_loss,dic_results,save_folder,trial_id):
    df_results,df_metrics,df_metrics_by_folds =  get_model_metrics(trainer,args,valid_losses,training_mode_list,metric_list,dic_results)
    if not os.path.exists(f"{SAVE_DIRECTORY}/{save_folder}"):
        os.makedirs(f"{SAVE_DIRECTORY}/{save_folder}")
    print('Saving results in: ',f"{SAVE_DIRECTORY}/{save_folder}/")
    df_results.to_csv(f"{SAVE_DIRECTORY}/{save_folder}/VALID_{trial_id}.csv")
    df_loss.to_csv(f"{SAVE_DIRECTORY}/{save_folder}/Losses_{trial_id}.csv")
    df_metrics.to_csv(f"{SAVE_DIRECTORY}/{save_folder}/METRICS_{trial_id}.csv")
    df_metrics_by_folds.to_csv(f"{SAVE_DIRECTORY}/{save_folder}/METRICS_BY_FOLD{trial_id}.csv")

    #print('df metrics: ',df_metrics)


def train_model_on_k_fold_validation(trial_id,load_config,save_folder,modification={},add_name_id=''):
    '''
    1. Load the best config according to our HP-Tuning
    2. Apply the K-fold validation to split inputs
    3. For each fold, load a new model and train it with the associated fold of inputs
    4. Keep track on train/valid losses and the best results. 
    5. Save them.
    '''
    # 1. Load the best config according to our HP-Tuning / Or Load random config :
    args,folds = load_configuration(trial_id,load_config)

    trial_id = f"{trial_id}{add_name_id}"
    # 2. 3. 4. 
    trainer,args,valid_losses,training_mode_list,metric_list,df_loss,dic_results = train_valid_K_models(args,trial_id,save_folder,modification)
    # 5.
    save_model_metrics(trainer,args,valid_losses,training_mode_list,metric_list,df_loss,dic_results,save_folder,trial_id)


# ========================================================
# Application 
# ========================================================
if __name__ == '__main__':
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
    if False: 
        save_folder = 'K_fold_validation/traing_without_HP_tuning'
        if not os.path.exists(f"{SAVE_DIRECTORY}/{save_folder}"):
            os.mkdir(f"{SAVE_DIRECTORY}/{save_folder}")

        load_config = False
        trial_id = 'train_random_config'
        epochs = 200
        train_model_on_k_fold_validation(trial_id,load_config,save_folder,epochs,hp_tuning_on_first_fold=False)
    # Case 2. We just need to test some configuration, where we set the configuration from 'load_random_config.py:

    # ----------
    if True: 
        save_folder = 'K_fold_validation/training_with_HP_tuning'
        load_config = True
        trial_id = 'subway_in_calendar_embedding_RNN_HuberLossLoss_2025_07_04_21_04_52162'
        epochs = 1000
        train_model_on_k_fold_validation(trial_id,load_config,save_folder,modification={'epochs':epochs},add_name_id='')


