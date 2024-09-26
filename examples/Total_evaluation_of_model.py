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
from examples.HP_parameter_choice import hyperparameter_tuning
from examples.train_model_on_k_fold_validation import train_model_on_k_fold_validation



def HP_and_valid_one_config(args,coverage,dataset_names,epochs_HP_tuning,epochs_validation,vision_model_name):
    # HP Tuning
    args.epochs = epochs_HP_tuning
    analysis,trial_id = hyperparameter_tuning(args,coverage,dataset_names,vision_model_name)

    # K-fold validation with best config: 
    train_model_on_k_fold_validation(trial_id,load_config=True,save_folder='K_fold_validation',epochs=epochs_validation,folder = 'save/HyperparameterTuning')



if __name__ == '__main__':

    from file00 import *

    model_name = 'STGCN' #'CNN'
    args,coverage = get_args_coverage(model_name)

    # Modification :
    args.K_fold = 6
    args.ray = True
    #args.device = 'cuda:0'
    
    # Init 
    epochs_HP_tuning = 100 
    epochs_validation = 500
    args.vision_input_type = 'unique_image_through_lyon' # 'image_per_stations' # 'unique_image_through_lyon'
    dataset_names = ['subway_in'] # ['calendar','netmob'] #['subway_in','netmob','calendar']
    vision_model_name = 'FeatureExtractorEncoderDecoder'  # 'ImageAvgPooling'  # 'FeatureExtractor_ResNetInspired_bis'  #'FeatureExtractor_ResNetInspired' #'MinimalFeatureExtractor',
    # 'AttentionFeatureExtractor' # 'FeatureExtractorEncoderDecoder' # 'VideoFeatureExtractorWithSpatialTemporalAttention'


    # HP and evaluate K-fold best config
    HP_and_valid_one_config(args,coverage,dataset_names,epochs_HP_tuning,epochs_validation,vision_model_name)
