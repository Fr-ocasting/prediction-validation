# GET PARAMETERS
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
from examples.train_and_visu_non_recurrent import evaluate_config,train_the_config,get_ds
from high_level_DL_method import load_optimizer_and_scheduler
from dl_models.full_model import full_model
from trainer import Trainer
from utils.loger import LOG
from utils.rng import set_seed
from examples.train_model import main 
from examples.train_model_on_k_fold_validation import save_model_metrics,get_conditions,keep_track_on_metrics,init_metrics
from constants.paths import SAVE_DIRECTORY, FOLDER_PATH

modification_compute = {'num_workers' : 4, # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
                            'persistent_workers' : True ,# False 
                            'pin_memory' : True ,# False 
                            'prefetch_factor' : 4, # None, 2,3,4,5 ... 
                            'drop_last' : False,  # True
                            'mixed_precision' : False, # True # False
                            'torch_compile' : 'compile', # 'compile' # 'jit_script' #'trace' # False
                            'device': torch.device('cuda:1')
                            }
modifications = {
    '10p': {'train_pourcent' : 10},
    # '15p': {'train_pourcent' : 15},
    # '25p':  {'train_pourcent' : 25},
    # '35p':  {'train_pourcent' : 35},
    '50p':  {'train_pourcent' : 50},
    # '75p':  {'train_pourcent' : 75},
    # '80p':  {'train_pourcent' : 80},
    # '85p':  {'train_pourcent' : 85},
    '90p':  {'train_pourcent' : 90},
    # '95p':  {'train_pourcent' : 95},
    #'100p': {'train_pourcent': 100},
                 }

if __name__ == "__main__":
    loger = LOG()
    target_data = 'PeMS08_flow'#'PeMS08_flow'#'CRITER_3_4_5_lanes_flow' #'subway_in'  # PeMS03 # PeMS04 # PeMS07 # PeMS08 # METR_LA # criter
    dataset_names = ['PeMS08_flow'] # ['CRITER_3_4_5_lanes_flow','netmob_POIs']  #['CRITER_3_4_5_lanes_flow']#['PeMS08_flow','PeMS08_occupancy','PeMS08_speed'] # ['subway_in','calendar_embedding'] #['PeMS03'] #['subway_in'] ['subway_in','subway_indiv'] #["subway_in","subway_out"] # ['subway_in','netmob_POIs_per_station'],["subway_in","subway_out"],["subway_in","calendar"] # ["subway_in"] # ['data_bidon'] # ['METR_LA'] # ['PEMS_BAY']
    dataset_for_coverage =['PeMS08_flow'] # ['CRITER_3_4_5_lanes_flow','netmob_POIs'] #['PeMS08_flow'] # ['subway_in','netmob_image_per_station']#['subway_in','subway_indiv'] # ['subway_in','netmob_image_per_station'] #  ['data_bidon','netmob'] #  ['subway_in','netmob']  # ['METR_LA'] # ['PEMS_BAY']


    
    for model_name in ['STAEformer','STGCN']: # ,

        subfolder = f'comparison_accuracy_per_size_{model_name}'
        config_file = importlib.import_module(f"constants.config_by_datasets.{target_data}.{model_name}")
        importlib.reload(config_file)
        modification = config_file.config
        SEED = config_file.SEED

        modification.update(modification_compute)

        for add_trial_id,modification_i in modifications.items():
            
            ## Set save path :
            trial_id = f"{target_data}_{add_trial_id}"
            
            weights_save_folder = f"K_fold_validation/training_wo_HP_tuning"
            save_folder = f"{weights_save_folder}/{subfolder}/{trial_id}"
            save_folder_with_root = f"{os.path.expanduser('~')}/prediction-validation/{SAVE_DIRECTORY}/{save_folder}"
            print(f"Save folder: {save_folder_with_root}")
            if not os.path.exists(save_folder_with_root):
                os.makedirs(save_folder_with_root)
            ## ..

            set_seed(SEED)

            modification.update(modification_i)
            args_init = local_get_args(model_name,
                            args_init = None,
                            dataset_names=dataset_names,
                            dataset_for_coverage=dataset_for_coverage,
                            modification = modification)

            
            fold_to_evaluate=[args_init.K_fold-1]

            trainer,ds,model,args = main(fold_to_evaluate,weights_save_folder,args_init,modification,trial_id)

            condition1,condition2,fold = get_conditions(args,fold_to_evaluate,[ds])
            valid_losses,df_loss,training_mode_list,metric_list,dic_results= init_metrics(args)
            df_loss, valid_losses,dic_results = keep_track_on_metrics(trainer,args,df_loss,valid_losses,dic_results,fold_to_evaluate,fold,condition1,condition2,training_mode_list,metric_list)

            save_model_metrics(trainer,args,valid_losses,training_mode_list,metric_list,df_loss,dic_results,save_folder,trial_id)
            test_metrics = trainer.performance['test_metrics']

            loger.add_log(test_metrics,['rmse','mae','mape','mse'],trial_id, args.step_ahead,args.horizon_step)

        loger.display_log()