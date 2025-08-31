import sys
import os

# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np 
from constants.config import get_args,update_modif
from pipeline.utils.save_results import get_date_id
from pipeline.K_fold_validation.K_fold_validation import KFoldSplitter
from pipeline.high_level_DL_method import load_model,load_optimizer_and_scheduler
from pipeline.trainer import Trainer
import matplotlib.pyplot as plt 
def local_get_args(model_name,args_init,dataset_names,dataset_for_coverage,modification):
    # Load base args
    args = get_args(model_name,dataset_names,dataset_for_coverage)
    print(args.num_nodes)

    # Modification :
    for key,value in modification.items():
        setattr(args,key,value)
    args.W = 0
    args.K_fold = 6   # Means we will use the first fold for the Ray Tuning and the 4 other ones to get the metrics
    args.ray = False

    #  evaluation on the first fold only :
    hp_tuning_on_first_fold = True # True # False // if True, then we remove the first fold as we consid we used it for HP-tuning
    args.evaluate_complete_ds = True  # True # False // if True, then evaluation also on the entiere ds 

    # update each modif
    args = update_modif(args)
    
    # set number of folds to evaluate
    folds =  [0]

    if args_init is not None:
        args.args_vision = args_init.args_vision
        args.contextual_positions = args_init.contextual_positions
        args.time_embedding = args_init.time_embedding
        args.vision_input_type = args_init.vision_input_type

    return(args,folds,hp_tuning_on_first_fold)

def get_trial_id(args,vision_model_name=None):
    date_id = get_date_id()
    dataset_names = '_'.join(args.dataset_names)
    model_names = '_'.join([args.model_name,vision_model_name]) if vision_model_name is not None  else args.model_name
    trial_id =  f"{dataset_names}_{model_names}_{args.loss_function_type}Loss_{date_id}"
    return trial_id

def get_inputs(args,vision_model_name,folds):
    K_fold_splitter = KFoldSplitter(args,vision_model_name,folds)
    K_subway_ds,dic_class2rpz,_ = K_fold_splitter.split_k_fold()
    return(K_fold_splitter,K_subway_ds,dic_class2rpz)


def train_on_ds(model_name,ds,args,trial_id,save_folder,dic_class2rpz,df_loss):
    model = load_model(ds, args,dic_class2rpz)
    optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
    trainer = Trainer(ds,model,args,optimizer,loss_function,scheduler = scheduler,dic_class2rpz = dic_class2rpz,show_figure = False,trial_id = trial_id, fold=0,save_folder = save_folder)
    trainer.train_and_valid(mod = 1000,mod_plot = None) 
    df_loss[f"{model_name}_train_loss"] = trainer.train_loss
    df_loss[f"{model_name}_valid_loss"] = trainer.valid_loss

    return(trainer,df_loss)

def keep_track_on_model_metrics(df_results,model_name,performance,metrics):
    performance = trainer.performance
    dict_row = {'Model':[model_name],
                'Valid_loss':[performance['valid_loss']]
                }
    for metric in metrics:
        if (metric == 'PICP') or (metric == 'MPIW'):
            add_name = f'calib_{trainer.type_calib}_'
        else:
            add_name = ''
        dict_row.update({f'Valid_{add_name}{metric}':[performance['valid_metrics'][metric]],
                        f'Test_{add_name}{metric}':[performance['test_metrics'][metric]]
                        })

    row = pd.DataFrame(dict_row)
    df_results = pd.concat([df_results,row])
    return df_results

if __name__ == '__main__':
    target_data = 'subway_in'
    for dataset_names,vision_model_name in zip([['subway_in','netmob_POIs'],['subway_in']],['VariableSelectionNetwork',None]):
        # GET PARAMETERS
        #dataset_names = ['subway_in','netmob_POIs'] # ["subway_in","calendar"] # ["subway_in"] # ['data_bidon'] # ['METR_LA'] # ['PEMS_BAY']  # ['data_bidon','netmob_bidon'] #['netmob_POIs']
        dataset_for_coverage = ['subway_in','netmob_POIs'] #  ['data_bidon','netmob'] #  ['subway_in','netmob']  # ['METR_LA'] # ['PEMS_BAY'] # ['data_bidon','netmob_bidon'] #['netmob_POIs'] 
        #vision_model_name = 'VariableSelectionNetwork' # None # 'VariableSelectionNetwork'

        assert target_data in dataset_names, f'You are trying to predict {target_data} with only these data: {dataset_names}'
        save_folder = 'benchmark/fold0/'
        df_loss,df_results = pd.DataFrame(),pd.DataFrame()
        modification = {'epochs' : 200, #100,
                        }
        
        model_names = ['MTGNN','CNN','STGCN','LSTM','GRU','RNN'] #'DCRNN',
        print(f'\n>>>>Training {model_names[0]} on {dataset_names}')
        # Tricky but here we net to set 'netmob' so that we will use the same period for every combination
        args,folds,hp_tuning_on_first_fold = local_get_args(model_names[0],
                                                            None,
                                                            dataset_names=dataset_names,
                                                            dataset_for_coverage=dataset_for_coverage,
                                                            modification = modification)
        trial_id = get_trial_id(args,vision_model_name=vision_model_name)
        K_fold_splitter,K_subway_ds,dic_class2rpz = get_inputs(args,vision_model_name,folds)
        ds = K_subway_ds[0]

        trainer,df_loss = train_on_ds(model_names[0],ds,args,trial_id,save_folder,dic_class2rpz,df_loss)
        metrics = trainer.metrics
        df_results = keep_track_on_model_metrics(df_results,model_names[0],trainer,metrics)
        for model_name in model_names[1:]:  # benchamrk on all the other models, with the same input base['MTGNN','STGCN', 'CNN', 'DCRNN']
            print(f'\n>>>>Training {model_name} on {dataset_names}')
            args,folds,hp_tuning_on_first_fold = local_get_args(model_name,
                                                                args,
                                                                dataset_names=dataset_names,
                                                                dataset_for_coverage=dataset_for_coverage,
                                                                modification = modification)
            trial_id = get_trial_id(args,vision_model_name=vision_model_name)
            trainer,df_loss = train_on_ds(model_name,ds,args,trial_id,save_folder,dic_class2rpz,df_loss)
            metrics = trainer.metrics
            df_results = keep_track_on_model_metrics(df_results,model_name,trainer,metrics)

        print(df_results)
        df_loss[[f"{model}_valid_loss" for model in model_names]].plot()
        df_results.to_csv(f'{parent_dir}/save/results/{trial_id}.csv')
        plt.show()

