import pandas as pd
from utilities_DL import get_DataSet_and_invalid_dates,get_MultiModel_loss_args_emb_opts,load_init_trainer,load_prediction
from DL_class import MultiModelTrainer, Trainer
from config import get_args
from save_results import build_results_df
from paths import folder_path,file_name,get_save_directory
import time 
import torch
import argparse

from ray_config import get_ray_config
import ray 
from ray import tune 

# ==== GET PARAMETERS ====
# Load config
model_name = 'STGCN'  #'CNN'
args = get_args(model_name)
# Modification pour HyperParameterTuning:
args.time_embedding = True
args.loss_function_type =   'quantile' #'MSE' 
#args = get_args(model_name = model_name,learn_graph_structure = True)  # MTGNN




# Modification :
args.epochs = 500
args.K_fold = 6   # Means we will use the first fold for the Ray Tuning and the 5 other ones to get the metrics
if torch.cuda.is_available():
    args.device = 'cuda:0'
    args.batch_size = 256
else :
    args.device = 'cpu'
    args.batch_size = 32

args.single_station = False
args.ray = True



if args.loss_function_type == 'MSE':
    args.out_dim = 1
    args.alpha = None
    args.track_pi = False

else:
    args.out_dim = 2
    args.alpha = 0.1
    args.track_pi = True


    
print("!!! Prediction sur une UNIQUE STATION et non pas les 40 ") if args.single_station else None
print(f"!!! Loss function: {args.loss_function_type} ") 
print(f"Model: {args.model_name}, K_fold = {args.K_fold}") 
    
    
    
config = {"lr": tune.qloguniform(1e-4, 1e-1, 5e-5),
          "weight_decay" : tune.uniform(0.0005, 0.1),
          "momentum" : tune.uniform(0.80, 0.99),
          "dropout" : tune.uniform(0,0.9),
          "scheduler" : tune.choice([True,None]),

          "torch_scheduler_milestone": tune.qrandint(1, 100, 4),
          "torch_scheduler_gamma": tune.uniform(0.9, 0.999),
          "torch_scheduler_lr_start_factor": tune.uniform(0.1, 1), 
        }

config_embedding = {#'calendar_class' : tune.choice([1,2,3]),
                    'embedding_dim' : tune.choice([2,3,4]),
                    'multi_embedding' : tune.choice([True,False]),
                    #'TE_transfer' : tune.choice([True,False]),
                    }


config_stgcn = {"Kt" : tune.choice([2,3,4]),
                "stblock_num" : tune.choice([2,3,4]),
                "act_fun" : tune.choice(['glu','gtu']),
                "Ks" :  tune.choice([2,3]),
                "graph_conv_type" : tune.choice(['cheb_graph_conv','graph_conv']),
                "gso_type" : tune.choice(['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj']),
                "adj_type" : tune.choice(['adj','corr','dist'])
                }

if args.time_embedding:
    config.update(config_embedding)

if args.model_name == 'STGCN':
     config.update(config_stgcn)
        
        
## Hyper Parameter Tuning sur le Fold 0

def load_trainer(config,folder_path,file_name,args):

    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    Datasets,DataLoader_list,dic_class2rpz,nb_words_embedding,time_slots_labels,dic_rpz2class = load_init_trainer(folder_path,file_name,args)
    (loss_function,Model_list,Optimizer_list,Scheduler_list,args_embedding) = get_MultiModel_loss_args_emb_opts(args,nb_words_embedding,dic_class2rpz,n_vertex = len(Datasets[0].columns))
    dataset,dataloader,model,optimizer,scheduler = Datasets[0],DataLoader_list[0],Model_list[0],Optimizer_list[0],Scheduler_list[0]


    trainer = Trainer(dataset,model,dataloader,
                    args,optimizer,loss_function,scheduler = scheduler,
                    args_embedding=args_embedding,
                    save_dir = None,dic_class2rpz=dic_class2rpz)
    return(trainer)

def Train_with_tune(config):
    trainer = load_trainer(config,folder_path,file_name,args)
    result_df = trainer.train_and_valid()



ray_scheduler,ray_search_alg,resources_per_trial,num_gpus,max_concurrent_trials,num_cpus = get_ray_config(args)


if ray.is_initialized:
    ray.shutdown()
    ray.init(num_gpus=num_gpus,num_cpus=num_cpus)

analysis = tune.run(
        Train_with_tune,
        config=config,
        num_samples=200,  # Increase num_samples for more random combinations
        resources_per_trial = resources_per_trial,
        max_concurrent_trials = max_concurrent_trials,
        scheduler = ray_scheduler,
        search_alg = ray_search_alg,
    )

name_save = f"Htuning_ray_analysis_{args.model_name}_loss{args.loss_function_type}_TE_{args.time_embedding}"
analysis.results_df.to_csv(f'{name_save}.csv')