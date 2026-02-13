import pandas as pd  # if not, I get this error while running a .py from terminal: 
# ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found (required by /root/anaconda3/envs/pytorch-2.0.1_py-3.10.5/lib/python3.10/site-packages/pandas/_libs/window/aggregations.cpython-310-x86_64-linux-gnu.so)
import torch 
import numpy as np 
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True

# Relative path:
import sys 
import os 
current_path = os.path.abspath(os.path.dirname(__file__))
working_dir = os.path.abspath(os.path.join(current_path,'..','..','..'))
#parent_dir = f"{parent_dir}/prediction_validation/"  # A utiliser sur .ipynb notebook
if working_dir not in sys.path:
    sys.path.insert(0,working_dir)
# ...

# Personnal import 
from pipeline.MACARON.train_model_on_k_fold_validation import train_model_on_k_fold_validation
from pipeline.Subset_HPO.load_best_config import load_best_config_from_HPO
from experiences.loop_train_save_log import loop_train_save_log
from pipeline.utils.loger import LOG
loger = LOG()

init_save_folder = 'K_fold_validation/training_wo_HP_tuning/Exp1_subway_out'
trial_id = 'subway_in_calendar_GMAN_HuberLossLoss_2025_11_30_18_01_76652'

# -------- Modifications: --------
epochs_validation = 200
torch_compile = False # 'compile' # False
K_fold = 1
device = torch.device('cuda:1')
# --------------------------------

dic_configs = {}
num_heads = 4
head_dim = 16
for i,lr in enumerate([0.001,0.0005,0.0003,0.0001]):
    modification = {'epochs':epochs_validation,
                    'torch_compile':torch_compile,
                    'K_fold':K_fold,
                    'device':device,
                    'num_heads': num_heads,
                    'head_dim':head_dim,
                    'nb_STAttblocks':3,
                    'lr':0.001,
                    'weight_decay':1e-4,
                    'torch_scheduler': True,
                    # 'torch_scheduler_type': 'MultiStepLR',
                    # 'loss_function_type':'HuberLoss',
                    # 'torch_scheduler_milestone': [25, 45, 65],
                    # 'torch_scheduler_gamma':0.1,
                    'torch_scheduler': False,
                    'train_prop': 0.6,
                    'valid_prop': 0.2,
                    'test_prop': 0.2,
                    'dropout':0.1,
                    'loss_function_type':'HuberLoss',
                    'optimizer': 'adamw',
                    }
    args,folds = load_best_config_from_HPO(trial_id)
    for key in modification:
        setattr(args,key,modification[key])
    
    dic_configs.update({f'config_{i}':vars(args)})

print(dic_configs.keys())
loop_train_save_log(loger,dic_configs,init_save_folder = init_save_folder) 