# GET PARAMETERS
import os 
import sys
import torch 
import numpy as np 
import pandas as pd 
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True
# Get Parent folder : 

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from examples.benchmark import local_get_args
from examples.train_and_visu_non_recurrent import evaluate_config,train_the_config,get_ds
from high_level_DL_method import load_optimizer_and_scheduler
from dl_models.full_model import full_model
from trainer import Trainer
from constants.config_by_datasets import dic_config

# Init:
#['subway_indiv','tramway_indiv','bus_indiv','velov','criter']
target_data = 'PeMS08'  # PeMS03 # PeMS04 # PeMS07 # PeMS08 # METR_LA 
dataset_names = [target_data] 
dataset_for_coverage = [target_data]
model_name = 'STGCN'
# ...


config2update = {'METR_LA': {'adj_type': 'dist',
                             'freq': '5min',
                             'step_ahead':12,
                             'minmaxnorm':False,
                             'standardize':True,
                             'K_fold':1,
                             'H':12,#12,
                             'D':0, #0,
                             'W':0,
                             'enable_padding': False,

                             'batch_size': 50,
                             #'optimizer': 'adam',
                             'epochs':50,
                             'lr': 0.001,
                             'dropout': 0.0,
                             'weight_decay': 0.0001,  # Pas sur  +   Optimizer n'est pas Adam 

                             'train_prop':0.7,
                             'valid_prop':0.1,

                             'Kt': 3,
                             'Ks':3,
                             'stblock_num':2,
                             'temporal_h_dim':64,
                             'spatial_h_dim': 16,
                             'output_h_dim': 16,
                             'threshold':0.5,
                             'gso_type': 'sym_renorm_adj',
                             },

                 'PeMS03': {'adj_type': 'dist',
                             'freq': '5min',
                             'step_ahead':12,
                             'H':12,
                             'D':0,
                             'W':0,
                             },

                 'PeMS04': {'adj_type': 'dist',
                             'freq': '5min',
                             'step_ahead':12,
                             'H':12,
                             'D':0,
                             'W':0,},

                 'PeMS07': {'adj_type': 'dist',
                             'freq': '5min',
                             'step_ahead':12,
                             'H':12,
                             'D':0,
                             'W':0,
                             },

                 'PeMS08': {'adj_type': 'dist',    # config from https://dl.acm.org/doi/pdf/10.1145/3571285
                             'freq': '5min',
                             'step_ahead':12,
                             'minmaxnorm':False,
                             'standardize':True,
                             'K_fold':1,
                             'H':12,
                             'D':0,
                             'W':0,
                             'enable_padding': False,

                             'batch_size': 256,#32,
                             'optimizer': 'adam',
                             'epochs':50,
                             'lr': 0.001,
                             'weight_decay': 0.0001,  # Pas sur  +   Optimizer n'est pas Adam 

                            'train_prop':0.6,
                             'valid_prop':0.2,

                             'Kt': 3,
                             'Ks':3,
                             'stblock_num':2,
                             'temporal_h_dim':64,
                             'spatial_h_dim': 16,
                             'output_h_dim': 16,
                             'threshold':0.5,
                             'gso_type': 'sym_renorm_adj',

                            'num_workers' : 4, # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
                            'persistent_workers' : True ,# False 
                            'pin_memory' : True ,# False 
                            'prefetch_factor' : 4, # None, 2,3,4,5 ... 
                            'drop_last' : False,  # True
                            'mixed_precision' : False, # True # False

                             }}
# Modif 
modification = {'target_data': target_data, 
                'use_target_as_context': False,
                'data_augmentation': False,
                'scheduler': False,  # None
                #'torch_scheduler_milestone': 28.0, #5,
                #'torch_scheduler_gamma': 0.9958348861339396, # 0.997,
                #'torch_scheduler_lr_start_factor': 0.8809942312067847, # 1,

                'learnable_adj_matrix' : False, # True      
                'stacked_contextual': False, # True # False
                'temporal_graph_transformer_encoder': False,
                'compute_node_attr_with_attn' : False, # True ??
                }
                 
modification.update(config2update[target_data])

if target_data in dic_config.keys():
    modification.update(dic_config[target_data])


args_init = local_get_args(model_name,
                    args_init = None,
                    dataset_names=dataset_names,
                    dataset_for_coverage=dataset_for_coverage,
                    modification = modification)


def main():
    ds,args,trial_id,save_folder,df_loss = get_ds(modification=modification,args_init=args_init)
    model = full_model(ds, args).to(args.device)
    #model = torch.jit.script(model)
    #model = torch.compile(model, mode="default", fullgraph=True)
    optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
    trainer = Trainer(ds,model,args,optimizer,loss_function,scheduler = scheduler,show_figure = False,trial_id = trial_id, fold=0,save_folder = save_folder)
    trainer.train_and_valid(normalizer = ds.normalizer, mod = 1000,mod_plot = None) 
    return trainer,ds,model,args


if __name__ == "__main__":
    # Run the script
    trainer,ds,model,args = main()
    print("Training completed successfully.")
    performance = trainer.performance
    records = [
    {'Step ahead': h,
     'RMSE': np.sqrt(performance['test_metrics'][f'mse_h{h}']),
     'MAE': performance['test_metrics'][f'mae_h{h}'],
     'MAPE': performance['test_metrics'][f'mape_h{h}']}
    for h in range(1, 13)
        ]
    df = pd.DataFrame(records).set_index('Step ahead')
    print(df)
    print(df.mean())