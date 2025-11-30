# Homogenous spatial unit 


# ----------------------------------------------------------------
# Aim of the experience: 

# Evaluate the impact of different fusion strategis of contextual data (early fusion, late fusion)
# on the performance of a spatio-temporal model (STAEformer) for a multi-sources traffic prediction 
# task (Subway-Out) with homogenous spatial units (Subway stations in Lyon).
# ----------------------------------------------------------------



# ----------------------------------------------------------------
# Limits: 

# Consider only one simple type of fusion operation (concatenation)
# While several others are possible (sum, gating mechanism, impulsion...)

# Consider only one feature extractor model (Transformer-based

# But possibility are infinit. That's why we choose to propose a 
# first benchmark to serve as a basis for future works and which 
# can guide feature extractor model choice and fusion strategy design.
# 
# ----------------------------------------------------------------



# ----------------------------------------------------------------
# Detail of experience :

# Dataset: Subway traffic in Lyon (Subway-In, Subway-Out), Calendar data
# Model: STGCN
# Target: Subway-Out
# Contextual: Subway-In, Calendar
# Number of trials : 5 per configuration
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# # Configurations :
# 
# With early fusion : 
# ---
#   Shared Embedding
#   Independant Embedding

# With late fusion :
# ---
#   Simple Embedding
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# Outputs: 
# Performance metrics (MAE, RMSE, MASE) of gains of each configuration compared to the baseline (no contextual data) (+- std ?) 
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# Code: 

# GET PARAMETERS
import os 
import sys
import torch 
import importlib
import itertools
import copy 
import torch._dynamo as dynamo; dynamo.graph_break()
torch._dynamo.config.verbose=True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True
# Get Parent folder : 

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..','..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from pipeline.utils.loger import LOG
from experiences.common_parameter_STGCN import possible_target_kwargs, subway_possible_contextual_kwargs, bike_possible_contextual_kwargs, model_configurations
from experiences.common_parameter_STGCN import REPEAT_TRIAL
from experiences.compilation_parameter import compilation_modification
from experiences.loop_train_save_log import loop_train_save_log

loger = LOG()

# --- Init ---  (set horizon, freq, ...)
# Set seed : NO 

REPEAT_TRIAL  = 5 # 1 # 5
EPOCHS = 100  # 80  # 1
TORCH_COMPILE =  'compile' # 'compile' # 'compile'  # False

init_save_folder = 'K_fold_validation/training_wo_HP_tuning/Exp1_subway_out'
device = torch.device('cuda:1')

freq = '15min'  
horizons = [1,4]  #[1,4]
target_data = 'subway_out'  # 'subway_out'
L_contextual_dataset_names = [['subway_in'],[]] # [['subway_out'],[]]  # [[],['subway_in']]
dic_integration_stategies = {'early_fusion':['shared_embedding',
                                            # 'independant_embedding',
                                            # 's_proj_t_proj',
                                            #  'traffic_model_backbone',

                                             ],
                             'late_fusion':[
                                            'traffic_model_backbone',
                                            #  'simple_embedding'
                                             ],
                             
                             
                             }

model_name = 'STGCN' # 'STAEformer'
config_backbone_model = model_configurations[model_name]



# ------ Possible configurations :
dic_configs = {}
def add_config(dic_configs,contextual_kwargs,contextual_dataset_names,model_name,fusion_type,feature_extractor_type,add_name=''):
    dataset_names =  [target_data] +contextual_dataset_names+ ['calendar_embedding'] if contextual_kwargs is not None else  [target_data] + ['calendar_embedding']

    name_i = f"{model_name}_{'_'.join(dataset_names)}_{fusion_type}_{feature_extractor_type}" if contextual_kwargs is not None else f"{model_name}_{'_'.join(dataset_names)}"
    if add_name != '':
        name_i = f"{name_i}_{add_name}"
    name_i_end = f"_e{EPOCHS}_h{horizon}_bis{n_bis}"
    name_i = f"{name_i}_{name_i_end}"

    config_i =  {'target_data': target_data,
                'dataset_names': dataset_names,
                'model_name': model_name,
                'dataset_for_coverage': [target_data],
                'freq': freq,
                'horizon_step': horizon,
                'step_ahead': horizon,
                'target_kwargs' : {target_data: possible_target_kwargs[target_data]},
                'contextual_kwargs' : {data_i:contextual_kwargs for data_i in contextual_dataset_names} if contextual_kwargs is not None else {},
                'denoising_names':[],
                } 
    config_i.update(config_backbone_model)
    config_i.update(compilation_modification)


    config_i['device'] = device
    config_i['epochs'] = EPOCHS
    config_i['torch_compile'] = TORCH_COMPILE

    dic_configs[name_i] = config_i
    return dic_configs




for contextual_dataset_names in L_contextual_dataset_names: 
    # ---- Config Without Contextual Data : 
    if contextual_dataset_names == []:
        contextual_kwargs,fusion_type,feature_extractor_type = None,None,None

        for horizon in horizons:
            for n_bis in range(1,REPEAT_TRIAL+1): # range(1,6):
                dic_configs = add_config(dic_configs,contextual_kwargs,contextual_dataset_names,model_name,fusion_type,feature_extractor_type,)

    # ---- Config With Contextual Data :
    else: 
        for fusion_type, L_feature_extractor_type in dic_integration_stategies.items():
            for feature_extractor_type in L_feature_extractor_type:
                contextual_kwargs = subway_possible_contextual_kwargs[fusion_type][feature_extractor_type]
                for horizon in horizons:
                    for n_bis in range(1,REPEAT_TRIAL+1): # range(1,6):
                        dic_configs = add_config(dic_configs,contextual_kwargs,contextual_dataset_names,model_name,fusion_type,feature_extractor_type,)
                
print('\n------------------------------- CONFIGURATIONS ----------------------\n')
for key in list(dic_configs.keys()):
    print('   ',key)
print('\n---------------------------------------------------------------------\n')


# --- Evaluate configurations ---
loop_train_save_log(loger,dic_configs,init_save_folder = init_save_folder) 

                



