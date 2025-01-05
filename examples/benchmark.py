import sys
import os
import gc
# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np 
from constants.config import get_args,update_modif, modification_contextual_args
from utils.save_results import get_date_id
from K_fold_validation.K_fold_validation import KFoldSplitter
from high_level_DL_method import load_model,load_optimizer_and_scheduler
from utils.save_results import get_trial_id
from trainer import Trainer
import matplotlib.pyplot as plt 
import importlib
def local_get_args(model_name,args_init,dataset_names,dataset_for_coverage,modification):
    # Load base args
    args = get_args(model_name,dataset_names,dataset_for_coverage)
    args.ray = False

    #  evaluation on the first fold only :
    hp_tuning_on_first_fold = True # True # False // if True, then we remove the first fold as we consid we used it for HP-tuning
    args.evaluate_complete_ds = True  # True # False // if True, then evaluation also on the entiere ds 

    # Modification :
    for key,value in modification.items():
        setattr(args,key,value)
    

    # update each modif
    args = update_modif(args)
    
    # set number of folds to evaluate
    folds =  [0]

    if args_init is not None:
        args.args_vision = args_init.args_vision
        args.args_embedding = args_init.args_embedding 
        args.contextual_positions = args_init.contextual_positions
        args.vision_input_type = args_init.vision_input_type

    if model_name == 'STGCN':
        from dl_models.STGCN.load_config import load_blocks
        blocks = load_blocks(args.stblock_num,args.temporal_h_dim, args.spatial_h_dim,args.output_h_dim)
        args.blocks = blocks

    

    return(args,folds,hp_tuning_on_first_fold)


def get_inputs(args,folds):
    K_fold_splitter = KFoldSplitter(args,folds)
    K_subway_ds,args = K_fold_splitter.split_k_fold()
    return(K_fold_splitter,K_subway_ds,args)

def train_on_ds(ds,args,trial_id,save_folder,df_loss):
    model = load_model(ds, args)
    optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
    trainer = Trainer(ds,model,args,optimizer,loss_function,scheduler = scheduler,show_figure = False,trial_id = trial_id, fold=0,save_folder = save_folder)
    trainer.train_and_valid(mod = 1000,mod_plot = None) 
    df_loss[f"{args.model_name}_train_loss"] = trainer.train_loss
    df_loss[f"{args.model_name}_valid_loss"] = trainer.valid_loss

    return(trainer,df_loss)

def keep_track_on_model_metrics(trainer,df_results,model_name,performance,metrics):
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

def plot_gradient(trainer):
    gradient_metrics = trainer.gradient_metrics
    for module in gradient_metrics.keys():
        df_to_plot = pd.DataFrame()
        for metric_grad in gradient_metrics[module].keys():
            df_to_plot[f"{module}_{metric_grad}"] = gradient_metrics[module][metric_grad]
        if len(df_to_plot) > 0: 
            if len(df_to_plot)> 3:
                df_to_plot.iloc[3:].plot()
            else:
                df_to_plot.plot()
if __name__ == '__main__':

    for dataset_names,vision_model_name in zip([['subway_in','calendar']],[None]): # zip([['subway_in','netmob_POIs','calendar'],['subway_in']],['VariableSelectionNetwork',None]):
        # GET PARAMETERS
        #dataset_names = ['subway_in','netmob_POIs'] # ["subway_in","calendar"] # ["subway_in"] # ['data_bidon'] # ['METR_LA'] # ['PEMS_BAY']  # ['data_bidon','netmob_bidon'] #['netmob_POIs']
        dataset_for_coverage = ['subway_in','netmob_POIs'] #['subway_in','netmob_POIs'] #  ['data_bidon','netmob'] #  ['subway_in','netmob']  # ['METR_LA'] # ['PEMS_BAY'] # ['data_bidon','netmob_bidon'] #['netmob_POIs'] 
        #vision_model_name = 'VariableSelectionNetwork' # None # 'VariableSelectionNetwork'

        from constants.paths import DATA_TO_PREDICT

        save_folder = 'benchmark/fold0/'
        df_loss,df_results = pd.DataFrame(),pd.DataFrame()
        modification = {'epochs' : 2, #100,
                        'lr':4e-4,
                        'set_spatial_units' : ['BON','SOI','GER','CHA'],
                        'TE_concatenation_early':True,
                        'TE_concatenation_late':False,
                        'TE_embedding_dim':4,
                        'TE_variable_selection_model_name': 'GRN'
                        }
    
        model_names =['STGCN'] # [None] # ['CNN','LSTM','GRU','RNN','STGCN'] #'DCRNN','MTGNN'
        print(f'\n>>>>Training {model_names[0]} on {dataset_names}')
        # Tricky but here we net to set 'netmob' so that we will use the same period for every combination
        args,folds,hp_tuning_on_first_fold = local_get_args(model_names[0],
                                                            args_init = None,
                                                            dataset_names=dataset_names,
                                                            dataset_for_coverage=dataset_for_coverage,
                                                            modification = modification)
        K_fold_splitter,K_subway_ds,args = get_inputs(args,folds)
        args = modification_contextual_args(args,modification)
        trial_id = get_trial_id(args)
        ds = K_subway_ds[0]

        trainer,df_loss = train_on_ds(ds,args,trial_id,save_folder,df_loss)
        metrics = trainer.metrics
        df_results = keep_track_on_model_metrics(trainer,df_results,model_names[0],trainer,metrics)

        # Display Gradient : 
        plot_gradient(trainer)

        for model_name in model_names[1:]:  # benchamrk on all the other models, with the same input base['MTGNN','STGCN', 'CNN', 'DCRNN']
            print(f'\n>>>>Training {model_name} on {dataset_names}')
            args,folds,hp_tuning_on_first_fold = local_get_args(model_name,
                                                                args_init = args,
                                                                dataset_names=dataset_names,
                                                                dataset_for_coverage=dataset_for_coverage,
                                                                modification = modification)
            trial_id = get_trial_id(args)
            trainer,df_loss = train_on_ds(ds,args,trial_id,save_folder,df_loss)
            metrics = trainer.metrics
            df_results = keep_track_on_model_metrics(trainer,df_results,model_name,trainer,metrics)

        print(df_results)
        df_loss[[f"{model}_valid_loss" for model in model_names]].plot()
        plt.show()
        df_results.to_csv(f'{parent_dir}/save/results/{trial_id}.csv')

        del args 
        del ds
        del K_fold_splitter
        del K_subway_ds
        del trainer 
        del df_results
        gc.collect()

       

