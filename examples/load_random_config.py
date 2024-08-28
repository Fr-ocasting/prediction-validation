import sys
import os

# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from utils.utilities_DL import match_period_coverage_with_netmob

from constants.config import get_args,update_modif
from constants.paths import file_name


trial_id = ''
# ==== GET PARAMETERS ====
model_name ='DCRNN' #'MTGNN' # 'STGCN'  #'CNN' # 
args = get_args(model_name)

# Modification :
args.epochs = 10
args.W = 0
args.K_fold = 5   # Means we will use the first fold for the Ray Tuning and the 4 other ones to get the metrics
args.ray = False
args.loss_function_type = 'quantile'  #'MSE' #

update_modif(args)

coverage = match_period_coverage_with_netmob(file_name)

# Choose DataSet and VisionModel if needed: 
dataset_names = ['subway_in'] # ['calendar','netmob'] #['subway_in','netmob','calendar']
vision_model_name = 'FeatureExtractor_ResNetInspired'  # 'ImageAvgPooling'  #'FeatureExtractor_ResNetInspired' #'MinimalFeatureExtractor',