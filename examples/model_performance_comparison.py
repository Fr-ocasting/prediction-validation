import sys
import os

# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np 
from train_model_on_k_fold_validation import train_valid_K_models,save_model_metrics
from utils.utilities_DL import match_period_coverage_with_netmob
from constants.config import get_args,update_modif
from constants.paths import FILE_NAME,SAVE_DIRECTORY
from utils.save_results import get_date_id

# ==== GET PARAMETERS ====
model_name ='STGCN' #'MTGNN' # 'STGCN'  #'CNN' # 'DCRNN'
args = get_args(model_name)

# Modification :
args.epochs = 500
args.W = 0
args.K_fold = 6   # Means we will use the first fold for the Ray Tuning and the 4 other ones to get the metrics
args.ray = False
args.loss_function_type = 'MSE'  #'MSE' # 'quantile'
args.scheduler = None

#  evaluation on the first fold only :
hp_tuning_on_first_fold = True # True # False // if True, then we remove the first fold as we consid we used it for HP-tuning
args.evaluate_complete_ds = True  # True # False // if True, then evaluation also on the entiere ds 
folds =  list(np.arange(args.K_fold)) #  list(np.arange(args.K_fold))  # [0]
# ...

update_modif(args)
coverage = match_period_coverage_with_netmob(FILE_NAME)


# Choose DataSet and VisionModel if needed: 
#dataset_names = ['subway_in'] # ['calendar','netmob'] #['subway_in','netmob','calendar']
vision_model_name = 'FeatureExtractor_ResNetInspired'  # 'ImageAvgPooling'  #'FeatureExtractor_ResNetInspired' #'MinimalFeatureExtractor',
save_folder = 'K_fold_validation/training_without_HP_tuning'

init_train_prop,init_valid_prop,init_test_prop = args.train_prop,args.valid_prop,args.test_prop
for dataset_names in [["subway_in","calendar"],["subway_in"]]:
    for torch_scheduler in [True, True]:
        for lr in [5e-3,5e-3]:
            for Kt in [3,2]:
                for stblock_num in [3,3]:
                    args.scheduler = torch_scheduler
                    args.stblock_num = stblock_num
                    args.lr = lr
                    args.Kt = Kt
                    print(f"\nModel perf on {dataset_names} with {model_name}")

                    # Keep good proportion even if we temporally change them during the process
                    args.train_prop,args.valid_prop,args.test_prop = init_train_prop,init_valid_prop,init_test_prop


                    date_id = get_date_id()
                    datasets_names = '_'.join(dataset_names)
                    model_names = '_'.join([args.model_name,vision_model_name]) if 'netmob' in dataset_names  else args.model_name
                    trial_id =  f"{datasets_names}_{model_names}_{args.loss_function_type}Loss_{date_id}"

                    trainer,args,valid_losses,training_mode_list,metric_list,df_loss = train_valid_K_models(dataset_names,args,coverage,vision_model_name,folds,hp_tuning_on_first_fold = hp_tuning_on_first_fold,trial_id = trial_id,save_folder=save_folder)

                    save_model_metrics(trainer,args,valid_losses,training_mode_list,metric_list,df_loss,save_folder,trial_id)

    