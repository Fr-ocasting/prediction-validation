# GET PARAMETERS
import os 
import sys
import torch 
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True
# Get Parent folder : 

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from examples.benchmark import local_get_args
from examples.train_and_visu_non_recurrent import evaluate_config,train_the_config,get_ds
from high_level_DL_method import load_optimizer_and_scheduler
from dl_models.full_model import full_model
from trainer import Trainer
# Init:
#['subway_indiv','tramway_indiv','bus_indiv','velov','criter']
target_data = 'subway_in'#'CRITER_3_4_5_lanes_flow' #'subway_in'  # PeMS03 # PeMS04 # PeMS07 # PeMS08 # METR_LA # criter
dataset_names = ['subway_in','calendar'] #['CRITER_3_4_5_lanes_flow']#['PeMS08_flow','PeMS08_occupancy','PeMS08_speed'] # ['subway_in','calendar_embedding'] #['PeMS03'] #['subway_in'] ['subway_in','subway_indiv'] #["subway_in","subway_out"] # ['subway_in','netmob_POIs_per_station'],["subway_in","subway_out"],["subway_in","calendar"] # ["subway_in"] # ['data_bidon'] # ['METR_LA'] # ['PEMS_BAY']
dataset_for_coverage = ['subway_in','netmob_image_per_station']#['CRITER_3_4_5_lanes_flow'] #['PeMS08'] # ['subway_in','netmob_image_per_station']#['subway_in','subway_indiv'] # ['subway_in','netmob_image_per_station'] #  ['data_bidon','netmob'] #  ['subway_in','netmob']  # ['METR_LA'] # ['PEMS_BAY']
model_name = 'STAEformer' # 'STGCN', 'ASTGCN' # 'STGformer' #'STAEformer' # 'DSTRformer'
#station = ['BEL','PAR','AMP','SAN','FLA']# ['BEL','PAR','AMP','SAN','FLA']   # 'BON'  #'GER'
# ...

# Modif 
modifications = {'identical_config': {},
                 'fast_training': {'batch_size': 128},

                 'optimizer_adamw': {'optimizer': 'adamw',
                                     'batch_size': 128
                                     },

                 'MSE_loss': {'loss_function_type':'MSE',
                              'optimizer': 'adamw',
                              'batch_size': 128},

                'adapted_adaptive_dim': {'adaptive_embedding_dim': 32,
                                           'optimizer': 'adamw',
                                           'batch_size': 128,
                                           },

                'adapted_embedding_dim': {'adaptive_embedding_dim': 32,
                                          'input_embedding_dim': 12,
                                           'tod_embedding_dim': 12,
                                            'dow_embedding_dim': 12,
                                           'optimizer': 'adamw',
                                           'batch_size': 128,
                                           },
                'adaptive32_input_tod_dow_12_b128': {'adaptive_embedding_dim': 32,
                                          'input_embedding_dim': 12,
                                           'tod_embedding_dim': 12,
                                            'dow_embedding_dim': 12,
                                           'optimizer': 'adamw',

                                           'batch_size': 128,
                                             'lr': 0.004,
                                             'weight_decay': 0.007,
                                             'dropout': 0.2,
                                            'torch_scheduler_milestone': [20],
                                            'torch_scheduler_gamma':0.9925,
                                            'torch_scheduler_type': 'warmup',
                                            'torch_scheduler_lr_start_factor': 0.3,

                                           },
                'adaptive32_input_tod_dow_12_b16': {'adaptive_embedding_dim': 32,
                            'input_embedding_dim': 12,
                            'tod_embedding_dim': 12,
                            'dow_embedding_dim': 12,
                            'optimizer': 'adamw',

                            'batch_size': 16,
                                'lr': 0.004,
                                'weight_decay': 0.007,
                                'dropout': 0.2,
                            'torch_scheduler_milestone': [20],
                            'torch_scheduler_gamma':0.9925,
                            'torch_scheduler_type': 'warmup',
                            'torch_scheduler_lr_start_factor': 0.3,
                            },
                'adaptive20_input_tod_dow_12_b16': {'adaptive_embedding_dim': 20,
                            'input_embedding_dim': 12,
                            'tod_embedding_dim': 12,
                            'dow_embedding_dim': 12,
                            'optimizer': 'adamw',

                            'batch_size': 16,
                                'lr': 0.004,
                                'weight_decay': 0.007,
                                'dropout': 0.2,
                            'torch_scheduler_milestone': [20],
                            'torch_scheduler_gamma':0.9925,
                            'torch_scheduler_type': 'warmup',
                            'torch_scheduler_lr_start_factor': 0.3,
                            },
                }





compilation_modification = {'use_target_as_context': False,
                            'data_augmentation': False,
                            'stacked_contextual': True, # True # False
                            'temporal_graph_transformer_encoder': False,
                            'compute_node_attr_with_attn' : False,

                            'epochs' : 500, #100

                            'num_workers' : 4, # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
                            'persistent_workers' : True ,# False 
                            'pin_memory' : True ,# False 
                            'prefetch_factor' : 4, # None, 2,3,4,5 ... 
                            'drop_last' : False,  # True
                            'mixed_precision' : False, # True # False
                            'torch_compile' : False, # 'compile' # 'jit_sript'

                            'device': torch.device('cuda:1')
    }



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(fold_to_evaluate,save_folder,modification):
    ds,args,trial_id,save_folder,df_loss = get_ds(modification=modification,args_init=args_init,fold_to_evaluate=fold_to_evaluate)
    for key,value in vars(args).items():
        print(f"{key}: {value}")
    model = full_model(ds, args).to(args.device)
    optimizer,scheduler,loss_function = load_optimizer_and_scheduler(model,args)
    trainer = Trainer(ds,model,args,optimizer,loss_function,scheduler = scheduler,show_figure = False,trial_id = trial_id, fold=0,save_folder = save_folder)
    trainer.train_and_valid(normalizer = ds.normalizer, mod = 1000,mod_plot = None) 
    return trainer,ds,model,args

if __name__ == "__main__":
    import numpy as np 
    import random 
    from constants.paths import SAVE_DIRECTORY, FOLDER_PATH
    from examples.train_model_on_k_fold_validation import save_model_metrics,get_conditions,keep_track_on_metrics,init_metrics
    import importlib
    
    config_file = importlib.import_module(f"constants.config_by_datasets.{target_data}.{model_name}")
    importlib.reload(config_file)
    modification_init = config_file.config
    SEED = config_file.SEED


    subfolder = 'STAEformer_architecture'
    for trial_id,modification_i in modifications.items():
        modification_model = modification_init.copy()
        modification_model.update(modification_i)
        modification_model.update(compilation_modification)
        args_init = local_get_args(model_name,
                        args_init = None,
                        dataset_names=dataset_names,
                        dataset_for_coverage=dataset_for_coverage,
                        modification = modification_model)

        set_seed(SEED)


        # Run the script
        fold_to_evaluate=[args_init.K_fold-1]

       
        save_folder = f"K_fold_validation/training_wo_HP_tuning/{subfolder}/{trial_id}"
        if False: 
            save_folder = f"{os.path.expanduser('~')}/prediction-validation/{SAVE_DIRECTORY}/K_fold_validation/training_wo_HP_tuning/{subfolder}/{trial_id}"
            print(f"Save folder: {save_folder}")
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

        trainer,ds,model,args = main(fold_to_evaluate,save_folder,modification_model)

        condition1,condition2,fold = get_conditions(args,fold_to_evaluate,[ds])
        valid_losses,df_loss,training_mode_list,metric_list,dic_results= init_metrics(args)
        df_loss, valid_losses,dic_results = keep_track_on_metrics(trainer,args,df_loss,valid_losses,dic_results,fold_to_evaluate,fold,condition1,condition2,training_mode_list,metric_list)

        save_model_metrics(trainer,args,valid_losses,training_mode_list,metric_list,df_loss,dic_results,save_folder,trial_id)
        print("Training completed successfully.")