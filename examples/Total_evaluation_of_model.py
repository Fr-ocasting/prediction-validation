import pandas as pd  # if not, I get this error while running a .py from terminal: 
# ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found (required by /root/anaconda3/envs/pytorch-2.0.1_py-3.10.5/lib/python3.10/site-packages/pandas/_libs/window/aggregations.cpython-310-x86_64-linux-gnu.so)

# Relative path:
import sys 
import os 
current_path = os.path.abspath(os.path.dirname(__file__))
working_dir = os.path.abspath(os.path.join(current_path,'..'))
#parent_dir = f"{parent_dir}/prediction_validation/"  # A utiliser sur .ipynb notebook
if working_dir not in sys.path:
    sys.path.insert(0,working_dir)
# ...

# Personnal import 
from utils.utilities_DL import match_period_coverage_with_netmob
from constants.config import get_args,update_modif
from constants.paths import file_name
from examples.HP_parameter_choice import hyperparameter_tuning
from examples.train_model_on_k_fold_validation import train_model_on_k_fold_validation



def HP_and_valid_one_config(args,coverage,dataset_names,vision_model_name):
    # HP Tuning
    analysis,trial_id = hyperparameter_tuning(args,coverage,dataset_names,vision_model_name)

    # K-fold validation with best config: 
    train_model_on_k_fold_validation(trial_id,load_config=True,save_folder='K_fold_validation',epochs=epochs,folder = 'save/HyperparameterTuning')



if __name__ == '__main__':

    # Load config
    model_name = 'STGCN' #'CNN'
    args = get_args(model_name)

    # Modification : 
    args.K_fold = 5
    args.ray = True
    args.W = 0  # IMPORTANT AVEC NETMOB
    args.epochs = 100
    args.loss_function_type = 'MSE' # 'quantile'

    args = update_modif(args)
    coverage = match_period_coverage_with_netmob(file_name)
    
    args.batch_size = 64  #otherwise 128 if cuda.is_available()
    
    # Use Small ds for fast training: 
    #small_ds = False
    #args.quick_ds = False
    #(coverage,args) = get_small_ds(small_ds,coverage,args)

    # Choose DataSet and VisionModel if needed: 
    
    dataset_names = ['subway_in'] # ['calendar','netmob'] #['subway_in','netmob','calendar']
    args.vision_input_type = 'unique_image_through_lyon' # 'image_per_stations' # 'unique_image_through_lyon'
    vision_model_name = 'FeatureExtractor_ResNetInspired'    # 'ImageAvgPooling'  # 'FeatureExtractor_ResNetInspired_bis'  #'FeatureExtractor_ResNetInspired' #'MinimalFeatureExtractor', # 'AttentionFeatureExtractor'# 'FeatureExtractorEncoderDecoder' # 'VideoFeatureExtractorWithSpatialTemporalAttention'

    HP_and_valid_one_config(args,coverage,dataset_names,vision_model_name)

