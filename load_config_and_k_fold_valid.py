import pandas as pd
from utilities_DL import get_DataSet_and_invalid_dates,get_MultiModel_loss_args_emb_opts,load_init_trainer
from DL_class import MultiModelTrainer, Trainer
from config import get_args
from save_results import build_results_df
from paths import folder_path,file_name,get_save_directory
import time 
import torch
import argparse

#from ray_config import get_ray_config
#import ray 
#from ray import tune 

# ==== GET PARAMETERS ====
# Load config
model_name = 'STGCN'  #'CNN'
args = get_args(model_name)
#args = get_args(model_name = model_name,learn_graph_structure = True)  # MTGNN

# Modification :
args.epochs = 500
args.K_fold = 6   # Means we will use the first fold for the Ray Tuning and the 5 other ones to get the metrics
if torch.cuda.is_available():
    args.device = 'cuda:1'
    args.batch_size = 256
else :
    args.device = 'cpu'
    args.batch_size = 32

args.single_station = False
args.ray = False


def update_args_according_loss_function(args):
    if args.loss_function_type == 'MSE':
        args.out_dim = 1
        args.alpha = None
        args.type_calendar = 'tuple'

        if args.ray:
            args.ray_track_pi = False

    else:
        args.embedding_dim = 3
        args.calendar_class = 3
        args.position = 'input'
        args.specific_lr = False
        args.type_calendar = 'tuple'
        args.out_dim = 2
        args.alpha = 0.1
        if args.ray:
            args.ray_track_pi = True
    return(args)



folder_config = 'HyperparameterTuning/'

def update_args(csv_path,args):
    for key in config_columns:
        value = row[f"config/{key}"]
        if hasattr(args, key):  #si l'attribu existe
            setattr(args, key, value) #le changer 
    return(args)

def update_args_according_TE(args,TE):
    if TE == 'False':
        args.time_embedding = False
    else:
        args.time_embedding = True
    return(args)


def load_multimodeltrainer_and_train_it(args):
    results_df = pd.DataFrame()
    save_dir = get_save_directory(args)
    Datasets,DataLoader_list,dic_class2rpz,nb_words_embedding,time_slots_labels,dic_rpz2class = load_init_trainer(folder_path,file_name,args)
    (loss_function,Model_list,Optimizer_list,Scheduler_list,args_embedding) = get_MultiModel_loss_args_emb_opts(args,nb_words_embedding,dic_class2rpz,n_vertex = len(Datasets[0].columns))
    # Remove the first Fold 
    Model_list,Optimizer_list,DataLoader_list,Datasets,Scheduler_list = Model_list[1:],Optimizer_list[1:],DataLoader_list[1:],Datasets[1:],Scheduler_list[1:]
    multimodeltrainer = MultiModelTrainer(Datasets,Model_list,DataLoader_list,args,Optimizer_list,loss_function,scheduler_list = Scheduler_list,args_embedding=args_embedding,save_dir = save_dir,dic_class2rpz=dic_class2rpz)

    (results_by_fold,mean_picp,mean_mpiw,dict_last_from_mean_of_folds,dict_best_from_mean_of_folds) = multimodeltrainer.K_fold_validation(mod_plot = 10)
    results_by_fold.to_csv(f"{save_dir}results_by_fold.csv")

    # Save results 
    results_df = build_results_df(results_df,args, mean_picp,mean_mpiw,dict_last_from_mean_of_folds,dict_best_from_mean_of_folds)
    results_df.to_csv(f"{args.model_name}_{args.loss_function_type}_H{args.H}_D{args.D}_W{args.W}_E{args.epochs}_K_fold{args.K_fold}_Emb_dim{args.embedding_dim}FC1_17_8_FC2_8_4_save_results.csv")


for TE in ['False']:
    for loss in ['quantile','MSE']:
        # Read Tune Analysis - Keep the 3 best configs
        csv_path = f"{folder_config}Htuning_ray_analysis_STGCN_loss{loss}_TE_{TE}.csv"
        df_config = pd.read_csv(csv_path,index_col = 0).sort_values('Loss_model')[:3]
        config_columns = [col.split('/')[1] for col in df_config.columns if col.startswith('config/')]
        
        # Update args for the 3 best config 
        for idx,row in df_config.iterrows():  # Pour chacune des 3meilleurs config : 
            args = update_args(csv_path,args)
            args = update_args_according_loss_function(args)
            args = update_args_according_TE(args,TE)
            load_multimodeltrainer_and_train_it(args)
            