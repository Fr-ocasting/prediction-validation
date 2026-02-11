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
from constants.config import get_args,update_modif



if __name__ == '__main__':

    # Load config
    model_name = 'STGCN' #'CNN' # 'STGCN' # 'RNN' # 'LSTM' # 'GRU' # 'MTGNN' # 'DCRNN'
    
    dataset_names = ["subway_in"] # ["subway_in","calendar"] # ["subway_in"] # ['data_bidon']
    dataset_for_coverage = ['subway_in','netmob'] # ["subway_in"] , ["netmob"], # ['subway_in','netmob']
    args = get_args(model_name,dataset_names,dataset_for_coverage)
    # Modification : 
    args.K_fold = 6
    args.ray = True
    args.W = 0  # IMPORTANT AVEC NETMOB
    args.loss_function_type = 'MSE' # 'quantile'
    args.epoch = 50
    args = update_modif(args)

    # Choose DataSet and VisionModel if needed: 
    num_samples = 1000
    vision_model_name = None #'FeatureExtractor_ResNetInspired'  # 'ImageAvgPooling'  #'FeatureExtractor_ResNetInspired' #'MinimalFeatureExtractor',
    analysis,trial_id = HPO_fold0_MACARON(args,num_samples)
