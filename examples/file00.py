import pandas as pd  # if not, I get this error while running a .py from terminal: 
# ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found (required by /root/anaconda3/envs/pytorch-2.0.1_py-3.10.5/lib/python3.10/site-packages/pandas/_libs/window/aggregations.cpython-310-x86_64-linux-gnu.so)


# Relative path:
import sys 
import os 
current_file_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
#parent_dir = f"{parent_dir}/prediction_validation/"  # A utiliser sur .ipynb notebook
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...


from constants.paths import FOLDER_PATH,FILE_NAME
from constants.config import get_args,update_modif
from utils.utilities_DL import match_period_coverage_with_netmob,get_small_ds


try :
    from examples.Total_evaluation_of_model import HP_and_valid_one_config
except:
    print('HP tuning impossible')

# Load config
def get_args_coverage(model_name = 'STGCN',dataset_names = ['subway_in','netmob']): #'CNN'
    ''' 
    
    Args:
    -----
    dataset_names with netmob allow to set coverage period as the intersection with the coverage period from netmob.
    '''
    args = get_args(model_name,dataset_names)

    # Modification : 
    args.ray = False
    args.loss_function_type = 'MSE' # 'quantile'
    args.mixed_precision = True
    args.torch_compile = False
    args = update_modif(args)

    coverage = match_period_coverage_with_netmob(FILE_NAME,dataset_names)
    (coverage,args) = get_small_ds(False,coverage,args)  #small_ds = False
    return args,coverage
