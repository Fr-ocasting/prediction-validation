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
import itertools 
import copy 
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
from experiences.common_parameter import possible_target_kwargs, weather_possible_contextual_kwargs, model_configurations,feature_extractor_model_configurations
from experiences.common_parameter import REPEAT_TRIAL
from experiences.compilation_parameter import compilation_modification
from experiences.loop_train_save_log import loop_train_save_log

loger = LOG()

# --- Init ---  (set horizon, freq, ...)
# Set seed : NO 

init_save_folder = 'K_fold_validation/training_wo_HP_tuning/Exp4_15min'
device = torch.device('cuda:1')

freq = '15min' #'15min'  
horizons = [4] # [4]  #[1,4]
target_data = 'bike_out' 
contextual_dataset_names = ['subway_in_subway_out']

model_name = 'STAEformer'
config_backbone_model = model_configurations[model_name]
config_backbone_model['epochs'] = 50 #80
compilation_modification['torch_compile'] = 'compile' # 'compile'  # False 
compilation_modification['device'] = device
REPEAT_TRIAL  = 1 

dic_configs = {}


weather_contextual_kwargs = weather_possible_contextual_kwargs['early_fusion']['repeat_t_proj']


# ------ Possible configurations :
L_input_embedding_dim = [24,48] # [8,24]
L_adaptive_embedding_dim = [16,32] # [0,16,32] 
L_init_adaptive_query_dim = [0,24,48] #  [0,8,24]
L_contextual_input_embedding_dim = [24,48] # [8,24] # [8,24,32,48]

adp_initquery_inputemb = list(itertools.product(L_adaptive_embedding_dim,L_init_adaptive_query_dim,L_contextual_input_embedding_dim,L_input_embedding_dim))
# adp_initquery_inputemb = list(itertools.product(L_adaptive_embedding_dim,L_init_adaptive_query_dim,L_input_embedding_dim))
# ------

subway_possible_contextual_kwargs = {'late_fusion':{}}
for adp_emb, init_adp_q, context_input_emb,input_emb in adp_initquery_inputemb:
# for adp_emb, init_adp_q,input_emb in adp_initquery_inputemb:
    contextual_kwargs_i = copy.deepcopy(feature_extractor_model_configurations)
    contextual_kwargs_i['attn_kwargs']['adaptive_embedding_dim'] = adp_emb
    contextual_kwargs_i['attn_kwargs']['init_adaptive_query_dim'] = init_adp_q
    contextual_kwargs_i['attn_kwargs']['contextual_input_embedding_dim'] = context_input_emb
    contextual_kwargs_i['attn_kwargs']['input_embedding_dim'] = input_emb
    contextual_kwargs_i['attn_kwargs']['concatenation_late'] = True
    contextual_kwargs_i['attn_kwargs']['cross_attention'] = True
    contextual_kwargs_i['backbone_model'] = True

    name_i = f"CrossAttnBackBone_InEmb{input_emb}_ctxInEmb{context_input_emb}_adp{adp_emb}_adpQ{init_adp_q}"
    # name_i = f"CrossAttnBackBone_InEmb{input_emb}_adp{adp_emb}_adpQ{init_adp_q}"

    subway_possible_contextual_kwargs['late_fusion'][name_i] =  contextual_kwargs_i


print('\n------------------------------- CONFIGURATIONS ----------------------------------\n')
print(list(subway_possible_contextual_kwargs['late_fusion'].keys()))
print('\n---------------------------------------------------------------------------------\n')

if True: 
    for fusion_type, config_contextual_kwargs in subway_possible_contextual_kwargs.items():
        for feature_extractor_type, contextual_kwargs_i in config_contextual_kwargs.items():
       
            contextual_kwargs ={'subway_out':contextual_kwargs_i,
                                'subway_in':contextual_kwargs_i,
                                'subway_in_subway_out':contextual_kwargs_i,
                                'weather':weather_contextual_kwargs}
            
            if 'weather' not in contextual_dataset_names:
                contextual_kwargs.pop('weather',None)  
            if 'subway_in' not in contextual_dataset_names:
                contextual_kwargs.pop('subway_in',None)  
            if 'subway_out' not in contextual_dataset_names:
                contextual_kwargs.pop('subway_out',None)
            if 'subway_in_subway_out' not in contextual_dataset_names:
                contextual_kwargs.pop('subway_in_subway_out',None)  


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

                    dic_configs[name_i] = config_i




loop_train_save_log(loger,dic_configs,init_save_folder = init_save_folder) 

                


