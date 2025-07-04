import os 
import sys
import torch 
import importlib

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True
# Get Parent folder : 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from examples.benchmark import local_get_args
from utils.loger import LOG
from utils.rng import set_seed
from examples.train_model import main 
from constants.paths import SAVE_DIRECTORY, FOLDER_PATH
from examples.train_model_on_k_fold_validation import save_model_metrics,get_conditions,keep_track_on_metrics,init_metrics



compilation_modification = {
    'epochs': 100,
    'num_workers' : 4, # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
    'persistent_workers' : True ,# False 
    'pin_memory' : True ,# False 
    'prefetch_factor' : 4, # None, 2,3,4,5 ... 
    'drop_last' : False,  # True
    'mixed_precision' : False, # True # False
    'torch_compile' : False , # 'compile' # 'jit_script' #'trace' # False
    'unormalize_loss' : True, # False
    'device': torch.device('cuda:1')
    }


if __name__ == "__main__":

    target_data = 'subway_in' # 'PeMS08_flow' # 'CRITER_3_4_5_lanes_flow' #'subway_in'  # PeMS03 # PeMS04 # PeMS07 # PeMS08 # METR_LA # criter
    # 'ASTGCN','MTGNN','DSTRformer','STGformer',
    loger = LOG()
    for model_name in ["RNN"]: # 'RNN','LSTM','GRU','DCRNN','STGCN','STAEformer', 'DCRNN'
        print('>>> Tackle model:',model_name)


        # Get Config: 
        
        try: 
            config_path = f"constants.config_by_datasets.{target_data}.{model_name}.{target_data}_calendar"
            module = importlib.import_module(config_path)
        except:
            try: 
                config_path = f"constants.config_by_datasets.{target_data}.{model_name}.{target_data}_calendar_embedding"
                module = importlib.import_module(config_path)
            except:
                try: 
                    config_path = f"constants.config_by_datasets.{target_data}.{model_name}.{target_data}"
                    module = importlib.import_module(config_path)
                except: 
                    raise FileNotFoundError(f"Configuration file for {target_data} and {model_name} not found in {config_path}.")

        config = module.config
        config.update(compilation_modification)
        SEED = module.SEED
        # ...

        # Save Folder: 
        weights_save_folder = f"K_fold_validation/training_wo_HP_tuning/benchmark"
        subfolder = f'{target_data}_{model_name}'
        save_folder = f"{weights_save_folder}/{subfolder}" 

        args_init = local_get_args(model_name,
                        args_init = None,
                        dataset_names=config['dataset_names'],
                        dataset_for_coverage=config['dataset_for_coverage'],
                        modification = config)
        fold_to_evaluate=[args_init.K_fold-1]
        # ...

        # Train Model 
        trainer,ds,model,args = main(fold_to_evaluate  = fold_to_evaluate,
                                     save_folder = weights_save_folder,
                                     args_init = args_init,
                                     modification = {},
                                     trial_id =subfolder)
        # ...


        # Keep Track on Metrics : 
        log_final  = f"\n--------- Resume ---------\n"
        condition1,condition2,fold = get_conditions(args,fold_to_evaluate,[ds])
        valid_losses,df_loss,training_mode_list,metric_list,dic_results= init_metrics(args)
        df_loss, valid_losses,dic_results = keep_track_on_metrics(trainer,args,df_loss,valid_losses,dic_results,fold_to_evaluate,fold,condition1,condition2,training_mode_list,metric_list)
        
        save_model_metrics(trainer = trainer,
                           args = args,
                           valid_losses = valid_losses,
                           training_mode_list = training_mode_list,
                           metric_list = metric_list,
                           df_loss = df_loss,
                           dic_results = dic_results,
                           save_folder = save_folder,
                           trial_id = subfolder)
        test_metrics = trainer.performance['test_metrics']
        loger.add_log(test_metrics,['rmse','mae','mape','mse'],subfolder, args.step_ahead,args.horizon_step)
    loger.display_log()
    # ...
