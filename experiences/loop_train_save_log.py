import os 
import sys
import torch 
import importlib
import glob
import torch._dynamo as dynamo; dynamo.graph_break()
torch._dynamo.config.verbose=True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True
# Get Parent folder : 

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from constants.config import local_get_args
from constants.paths import SAVE_DIRECTORY
from pipeline.high_level_DL_method import model_loading_and_training 
from pipeline.MACARON.train_model_on_k_fold_validation import save_model_metrics,get_conditions,keep_track_on_metrics,init_metrics
from pipeline.utils.loger import LOG

def loop_train_save_log(loger,dic_configs,init_save_folder = 'K_fold_validation/training_wo_HP_tuning/optim'):
    for trial_id,config_i in dic_configs.items():
        print('\n---------------------------------------------\nSTART TRIAL ID:',trial_id)
        trainer,ds,model,args,loger = train_one_config(config_i,init_save_folder,trial_id,loger=loger)
    return loger


def train_one_config(config_i,init_save_folder,trial_id,loger = None, modification = {}):
    if loger is None:
        loger = LOG()
    for k,v in modification.items():
        config_i[k] = v
    target_data = config_i['target_data']
    model_name = config_i['model_name']
    subfolder = f'{target_data}_{model_name}'

    args_init = local_get_args(model_name,
                    args_init = None,
                    dataset_names=config_i['dataset_names'],
                    dataset_for_coverage=config_i['dataset_for_coverage'],
                    modification = config_i)
    fold_to_evaluate=[args_init.K_fold-1]


    # Run the script
    if init_save_folder is not None: 
        weights_save_folder = f"{init_save_folder}/{subfolder}"
        if trial_id is None:
            raise ValueError(f"trial_id must be provided when init_save_folder ({init_save_folder}) is not None")
        save_folder = f"{weights_save_folder}/{trial_id}"
        save_folder_with_root = f"{SAVE_DIRECTORY}/{save_folder}"
        print(f"    Save folder: {save_folder_with_root}")
        if not os.path.exists(f"{SAVE_DIRECTORY}/{init_save_folder}"):
            os.mkdir(f"{SAVE_DIRECTORY}/{init_save_folder}")
        if not os.path.exists(f"{SAVE_DIRECTORY}/{weights_save_folder}"):
            os.mkdir(f"{SAVE_DIRECTORY}/{weights_save_folder}")
        if not os.path.exists(save_folder_with_root):
            os.mkdir(save_folder_with_root)
    else:
        weights_save_folder = None
        save_folder = None

    # savoir si il existe un fichier qui commence par  f"{save_folder_with_root}/{trial_id}_f" et qui est déja sauvegardé
    if weights_save_folder is not None and os.path.exists(f"{save_folder_with_root}/METRICS_{trial_id}.csv"):
        print(f"    Trial ID: {trial_id} ALREADY DONE, SKIPPING...")
        return None,None,None,None,loger

    else:
        # Train Model
        trainer,ds,model,args = model_loading_and_training(fold_to_evaluate,
                                        save_folder = weights_save_folder,
                                        args_init = args_init,
                                        modification ={},
                                        trial_id = trial_id)


        condition1,condition2,fold = get_conditions(args,fold_to_evaluate,[ds])
        valid_losses,df_loss,training_mode_list,metric_list,dic_results= init_metrics(args)
        df_loss, valid_losses,dic_results = keep_track_on_metrics(trainer,
                                                                  args,
                                                                  df_loss, 
                                                                  valid_losses,
                                                                  dic_results,
                                                                  fold_to_evaluate,
                                                                  fold,
                                                                  condition1,
                                                                  condition2,
                                                                  training_mode_list,
                                                                  metric_list
                                                                  )

        save_model_metrics(trainer,args,valid_losses,training_mode_list,metric_list,df_loss,dic_results,save_folder,trial_id)
        test_metrics = trainer.performance['test_metrics']

        loger.add_log(test_metrics,['rmse','mae','mase','mape'],trial_id, args.step_ahead,args.horizon_step)
        loger.display_log()
        return trainer,ds,model,args,loger