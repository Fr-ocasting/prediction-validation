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
from constants.paths import FOLDER_PATH,FILE_NAME
from K_fold_validation.K_fold_validation import KFoldSplitter

# Hp Tuning
from HP_tuning.hyperparameter_tuning_ray import HP_tuning



# === Train and Evaluate Model: 
def hyperparameter_tuning(args,coverage,dataset_names,vision_model_name,num_samples = 100):
    # Load K-fold subway-ds 
    folds = [0] # Here we use the first fold for HP-tuning. In case we need to compute the Sliding K-fold validation: folds = np.arange(1,args.K_fold)

    # Split in K-fold : 
    K_fold_splitter = KFoldSplitter(dataset_names,args,coverage,vision_model_name,folds)
    K_subway_ds,dic_class2rpz,_ = K_fold_splitter.split_k_fold()

    # Train on the first fold: 
    subway_ds = K_subway_ds[0]
    analysis,trial_id = HP_tuning(subway_ds,args,num_samples,dic_class2rpz,working_dir)
    return(analysis,trial_id)


if __name__ == '__main__':

    # Load config
    model_name = 'STGCN' #'CNN'
    dataset_names = ['subway_in']
    args = get_args(model_name,dataset_names)
    # Modification : 
    args.K_fold = 5
    args.ray = True
    args.W = 0  # IMPORTANT AVEC NETMOB
    args.epochs = 100
    args.loss_function_type = 'MSE' # 'quantile'

    args = update_modif(args)

    coverage = match_period_coverage_with_netmob(FILE_NAME,dataset_names = ['calendar','netmob'])
    # Use Small ds for fast training: 
    #small_ds = False
    #(coverage,args) = get_small_ds(small_ds,coverage,args)

    # Choose DataSet and VisionModel if needed: 

    vision_model_name = 'FeatureExtractor_ResNetInspired'  # 'ImageAvgPooling'  #'FeatureExtractor_ResNetInspired' #'MinimalFeatureExtractor',

    analysis,trial_id = hyperparameter_tuning(args,coverage,dataset_names,vision_model_name)
