# Heterogenous spatial unit 


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
# Model: STAEformer
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
#   Feature Extractor

# With late fusion :
# ---
#   Simple Embedding
#   Feature Extraxtor
#   Traffic Model backbone
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
import torch._dynamo as dynamo; dynamo.graph_break()
torch._dynamo.config.verbose=True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True
# Get Parent folder : 

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from pipeline.utils.loger import LOG
from experiences.common_parameter import possible_target_kwargs, weather_possible_contextual_kwargs, subway_possible_contextual_kwargs, model_configurations
from experiences.common_parameter import REPEAT_TRIAL
from experiences.compilation_parameter import compilation_modification
from experiences.loop_train_save_log import loop_train_save_log

loger = LOG()

# --- Init ---  (set horizon, freq, ...)
# Set seed : NO 

init_save_folder = 'K_fold_validation/training_wo_HP_tuning/Exp3_bike_15min_h4'
device = torch.device('cuda:1')

freq = '15min' #'15min'  
horizons = [4] # [4]  #[1,4]
target_data = 'bike_out' 
contextual_dataset_names = ['subway_out','weather']

model_name = 'STAEformer'
config_backbone_model = model_configurations[model_name]


dic_configs = {}
# REPEAT_TRIAL  = 1 

# Exp i. Fusion Strategy on STAEformer 
# --- Create configurations to evaluate ---


"""
En théorie, pourrait rajouter "feature extractor" en early fusion. Mais pas utile de tester ici.

"""

weather_contextual_kwargs = weather_possible_contextual_kwargs['early_fusion']['repeat_t_proj']

if True: 
    for fusion_type, config_contextual_kwargs in subway_possible_contextual_kwargs.items():
        for feature_extractor_type, contextual_kwargs_i in config_contextual_kwargs.items():
            if (feature_extractor_type == 'shared_embedding'):
                continue
            if feature_extractor_type == 'feature_extractor':
                continue
            if feature_extractor_type == 'traffic_model_backbone':
                continue
            if feature_extractor_type == 'simple_embedding':
                continue
            if feature_extractor_type == 'independant_embedding':
                continue

            contextual_kwargs ={'subway_out':contextual_kwargs_i,
                                'subway_in':contextual_kwargs_i,
                                'weather':weather_contextual_kwargs}
            
            if 'weather' not in contextual_dataset_names:
                contextual_kwargs.pop('weather',None)  
            if 'subway_in' not in contextual_dataset_names:
                contextual_kwargs.pop('subway_in',None)  
            if 'subway_out' not in contextual_dataset_names:
                contextual_kwargs.pop('subway_out',None)


            for horizon in horizons:
                for n_bis in range(1,REPEAT_TRIAL+1): # range(1,6):
                    dataset_names =  [target_data] +contextual_dataset_names+ ['calendar']
                    name_i = f"{model_name}_{'_'.join(dataset_names)}_{fusion_type}_{feature_extractor_type}"
                    name_i_end = f"_e{config_backbone_model['epochs']}_h{horizon}_bis{n_bis}"
                    name_i = f"{name_i}_{name_i_end}"

                    config_i =  {'target_data': target_data,
                                'dataset_names': dataset_names,
                                'model_name': model_name,
                                'dataset_for_coverage': [target_data],
                                'freq': freq,
                                'horizon_step': horizon,
                                'step_ahead': horizon,
                                'target_kwargs' : {target_data: possible_target_kwargs[target_data]},
                                'contextual_kwargs' : contextual_kwargs,
                                'denoising_names':[],
                                } 
                    config_i.update(config_backbone_model)
                    config_i.update(compilation_modification)


                    config_i['device'] = device
                    config_i['torch_compile'] = False 
                    # config_i['epochs'] = 1

                    dic_configs[name_i] = config_i


if True:
    for horizon in horizons:
        for n_bis in range(1,REPEAT_TRIAL+1): # range(1,6):
            dataset_names =  [target_data] + ['calendar']
            name_i = f"{model_name}_{'_'.join(dataset_names)}"
            name_i_end = f"_e{config_backbone_model['epochs']}_h{horizon}_bis{n_bis}"
            name_i = f"{name_i}_{name_i_end}"

            config_i =  {'target_data': target_data,
                        'dataset_names': dataset_names,
                        'model_name': model_name,
                        'dataset_for_coverage': [target_data],
                        'freq': freq,
                        'horizon_step': horizon,
                        'step_ahead': horizon,
                        'target_kwargs' : {target_data: possible_target_kwargs[target_data]},
                        'contextual_kwargs' : {},
                        'denoising_names':[],
                        } 
            config_i.update(config_backbone_model)
            config_i.update(compilation_modification)


            config_i['device'] = device
            config_i['torch_compile'] = False 
            # config_i['epochs'] = 1

            dic_configs[name_i] = config_i



# Exp ii. Fusion Strategy on STGCN

# Already implemented: only 'shared_embedding' with early fusion or 'simple_embedding' with late fusion.
#   I don't wan't to use 'STAEformer or Attention module based on what I developped from STAEformer, cause 
#   it would consider extract features with a powerfull Transformer-based model which won't produce a fair
#   comparison with STGCN. 
#   Also, it will need to adapt the code and dimensions within STGCN, which can be tricky, and consume a lot of time. 

if False:
    model_name = 'STGCN'
    config_backbone_model = model_configurations[model_name]
    for fusion_type, config_contextual_kwargs in weather_possible_contextual_kwargs.items():
        for feature_extractor_type, contextual_kwargs in config_contextual_kwargs.items():
            if (feature_extractor_type == 'feature_extractor'):
                continue
            if (feature_extractor_type == 'traffic_model_backbone'):
                continue
            if (feature_extractor_type == 'independant_embedding'):
                continue
            for horizon in horizons:
                for n_bis in range(1,REPEAT_TRIAL+1): # range(1,6):
                    dataset_names =  [target_data] +contextual_dataset_names+ ['calendar_embedding']
                    name_i = f"{model_name}_{'_'.join(dataset_names)}_{fusion_type}_{feature_extractor_type}"
                    name_i_end = f"_e{config_backbone_model['epochs']}_h{horizon}_bis{n_bis}"
                    name_i = f"{name_i}_{name_i_end}"

                    config_i =  {'target_data': target_data,
                                'dataset_names': dataset_names,
                                'model_name': model_name,
                                'dataset_for_coverage': [target_data],
                                'freq': freq,
                                'horizon_step': horizon,
                                'step_ahead': horizon,
                                'target_kwargs' : {target_data: possible_target_kwargs[target_data]},
                                'contextual_kwargs' : {data_i:contextual_kwargs for data_i in contextual_dataset_names},
                                'denoising_names':[],
                                } 
                    config_i.update(config_backbone_model)
                    config_i.update(compilation_modification)


                    config_i['device'] = device
                    config_i['torch_compile'] = False 
                    # config_i['epochs'] = 1

                    dic_configs[name_i] = config_i
                    
# # Already done with exp1.
# if False:
#     for horizon in horizons:
#         for n_bis in range(1,REPEAT_TRIAL+1): # range(1,6):
#             dataset_names =  [target_data] + ['calendar_embedding']
#             name_i = f"{model_name}_{'_'.join(dataset_names)}"
#             name_i_end = f"_e{config_backbone_model['epochs']}_h{horizon}_bis{n_bis}"
#             name_i = f"{name_i}_{name_i_end}"

#             config_i =  {'target_data': target_data,
#                         'dataset_names': dataset_names,
#                         'model_name': model_name,
#                         'dataset_for_coverage': [target_data],
#                         'freq': freq,
#                         'horizon_step': horizon,
#                         'step_ahead': horizon,
#                         'target_kwargs' : {target_data: possible_target_kwargs[target_data]},
#                         'contextual_kwargs' : {},
#                         'denoising_names':[],
#                         } 
#             config_i.update(config_backbone_model)
#             config_i.update(compilation_modification)


#             config_i['device'] = device
#             config_i['torch_compile'] = False 
#             # config_i['epochs'] = 1

#             dic_configs[name_i] = config_i

# --- Evaluate configurations ---
loop_train_save_log(loger,dic_configs,init_save_folder = init_save_folder) 

                


