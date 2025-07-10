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
from utils.loger import LOG
from utils.rng import set_seed
from examples.train_model import main 
from constants.paths import SAVE_DIRECTORY, FOLDER_PATH
from examples.train_model_on_k_fold_validation import save_model_metrics,get_conditions,keep_track_on_metrics,init_metrics


if __name__ == "__main__":

    target_data = 'subway_in' # 'PeMS08_flow' # 'CRITER_3_4_5_lanes_flow' #'subway_in'  # PeMS03 # PeMS04 # PeMS07 # PeMS08 # METR_LA # criter
    
    for model_name in ['STGCN']: # ,'STAEformer' 'STGCN'
        loger = LOG()
        # Init:
        #['subway_indiv','tramway_indiv','bus_indiv','velov','criter']

        #dataset_names = ['subway_in','netmob_POIs'] # ['PeMS08_flow'] #['CRITER_3_4_5_lanes_flow']#['PeMS08_flow','PeMS08_occupancy','PeMS08_speed'] # ['subway_in','calendar_embedding'] #['PeMS03'] #['subway_in'] ['subway_in','subway_indiv'] #["subway_in","subway_out"] # ['subway_in','netmob_POIs_per_station'],["subway_in","subway_out"],["subway_in","calendar"] # ["subway_in"] # ['data_bidon'] # ['METR_LA'] # ['PEMS_BAY']
        #dataset_for_coverage = ['subway_in','netmob_POIs'] # ['PeMS08_flow']#['CRITER_3_4_5_lanes_flow'] #['PeMS08'] # ['subway_in','netmob_image_per_station']#['subway_in','subway_indiv'] # ['subway_in','netmob_image_per_station'] #  ['data_bidon','netmob'] #  ['subway_in','netmob']  # ['METR_LA'] # ['PEMS_BAY']

        #station = ['BEL','PAR','AMP','SAN','FLA']# ['BEL','PAR','AMP','SAN','FLA']   # 'BON'  #'GER'
        # ...

        if target_data == 'PeMS08_flow':
            if model_name == 'STGCN':
                from examples.reproductibility.config_STGCN_PeMS08 import modifications as modifications

        if target_data == 'subway_in':
            if model_name == 'STAEformer':
                from examples.reproductibility.config_STAEformer_Subway_in_NetMob_calendar import modifications as modifications

            if model_name == 'STGCN':
                from examples.reproductibility.config_STGCN_Subway_in_NetMob_calendar import modifications as modifications
        if target_data == 'CRITER_3_4_5_lanes_flow':
            if model_name == 'STGCN':
                from examples.reproductibility.config_STGCN_CRITER_3_4_5_lanes_flow import modifications as modifications

            if model_name == 'STAEformer':
                from examples.reproductibility.config_STAEformer_CRITER_3_4_5_lanes_flow import modifications as modifications


        compilation_modification = { 'epochs' : 1, #100

                                    'num_workers' : 4, # 0,1,2, 4, 6, 8 ... A l'IDRIS ils bossent avec 6 num workers par A100 80GB
                                    'persistent_workers' : True ,# False 
                                    'pin_memory' : True ,# False 
                                    'prefetch_factor' : 4, # None, 2,3,4,5 ... 
                                    'drop_last' : False,  # True
                                    'mixed_precision' : False, # True # False
                                    'torch_compile' : False, # 'compile' , # 'compile' # 'jit_script' #'trace' # False
                                    'unormalize_loss' : True, # False

                                    'device': torch.device('cuda:1')
            }
        
        SEED = 1
        modification_init = {}
        set_seed(SEED)


        log_final  = f"\n--------- Resume ---------\n"
        subfolder = f'{target_data}_{model_name}'
        for trial_id,modification_i in modifications.items():
            print('\n>>>>>>>>>>>> TRIAL ID:',trial_id)
            config = modification_init.copy()
            config.update(compilation_modification)
            config.update(modification_i)

            args_init = local_get_args(model_name,
                            args_init = None,
                            dataset_names=config['dataset_names'],
                            dataset_for_coverage=config['dataset_for_coverage'],
                            modification = config)
            fold_to_evaluate=[args_init.K_fold-1]



            # Run the script
            weights_save_folder = f"K_fold_validation/training_wo_HP_tuning/optim/{subfolder}"
            save_folder = f"{weights_save_folder}/{trial_id}"
            save_folder_with_root = f"{os.path.expanduser('~')}/prediction-validation/{SAVE_DIRECTORY}/{save_folder}"
            print(f"Save folder: {save_folder_with_root}")
            if not os.path.exists(save_folder_with_root):
                os.makedirs(save_folder_with_root)
                
            # Train Model
            trainer,ds,model,args = main(fold_to_evaluate,
                                         save_folder = weights_save_folder,
                                         args_init = args_init,
                                         modification =config,
                                         trial_id = trial_id)
        

            condition1,condition2,fold = get_conditions(args,fold_to_evaluate,[ds])
            valid_losses,df_loss,training_mode_list,metric_list,dic_results= init_metrics(args)
            df_loss, valid_losses,dic_results = keep_track_on_metrics(trainer,args,df_loss,valid_losses,dic_results,fold_to_evaluate,fold,condition1,condition2,training_mode_list,metric_list)

            save_model_metrics(trainer,args,valid_losses,training_mode_list,metric_list,df_loss,dic_results,save_folder,trial_id)
            test_metrics = trainer.performance['test_metrics']

            loger.add_log(test_metrics,['rmse','mae','mape','mse'],trial_id, args.step_ahead,args.horizon_step)

        loger.display_log()