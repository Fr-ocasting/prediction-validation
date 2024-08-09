
import os
import sys

# Add parent path for package work and change current working dictory for future save: 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
     sys.path.insert(0, parent_dir)
#os.chdir(parent_dir)


from constants.paths import folder_path,file_name
from constants.config import get_args,update_modif
from high_level_DL_method import evaluate_config 
from utils.utilities_DL import match_period_coverage_with_netmob,get_small_ds


# Load config
model_name = 'STGCN' #'CNN'
args = get_args(model_name)

# Modification : 
args.K_fold = 1
args.epochs = 1
args.loss_function_type = 'MSE' # 'quantile'

args = update_modif(args)

# Coverage Period : 
small_ds = False
coverage = match_period_coverage_with_netmob(file_name)
(coverage,args) = get_small_ds(small_ds,coverage,args)

# Choose DataSet and VisionModel if needed: 
dataset_names = ['netmob'] # ['calendar','netmob'] #['subway_in','netmob','calendar']
vision_model_name = 'ImageAvgPooling'  # 'ImageAvgPooling'  #'FeatureExtractor_ResNetInspired' #'MinimalFeatureExtractor',

# Train and Evaluate Model: 
mod_plot = 1 # bokeh plotting every epoch 
trainer,model,args,pi,pi_cqr = evaluate_config(dataset_names,folder_path,file_name,args,coverage,vision_model_name,mod_plot)