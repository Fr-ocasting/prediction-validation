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
from experiences.common_parameter import possible_target_kwargs, subway_possible_contextual_kwargs, bike_possible_contextual_kwargs, model_configurations
from experiences.common_parameter import REPEAT_TRIAL
from experiences.compilation_parameter import compilation_modification
from experiences.loop_train_save_log import loop_train_save_log

loger = LOG()

# --- Init ---  (set horizon, freq, ...)
# Set seed : NO 

init_save_folder = 'K_fold_validation/training_wo_HP_tuning/Exp1_subway_out'
device = torch.device('cuda:1')

freq = '15min'  
horizons = [1,4]  #[1,4]
target_data = 'subway_out'  # 'subway_out'
L_contextual_dataset_names = [[],['subway_in']] # [['subway_out'],[]]  # [[],['subway_in']]
dic_integration_stategies = {'early_fusion':['s_proj_t_proj','shared_embedding','independant_embedding',
                                             'traffic_model_backbone',
                                             'adp_query_cross_attn_traffic_model_backbone',

                                             ],
                             'late_fusion':[
                                             'adp_query_cross_attn_traffic_model_backbone',
                                             'traffic_model_backbone','simple_embedding'
                                             ],
                             
                             
                             }

model_name = 'STAEformer'
config_backbone_model = model_configurations[model_name]



# ------ Possible configurations :
L_input_embedding_dim = [24] # [24,48] # [8,24]
L_adaptive_embedding_dim = [16] # [16,32] # [0,16,32] 
L_init_adaptive_query_dim = [0,24] #  [0,24,48] #  [0,8,24]
L_contextual_input_embedding_dim = [8,24] #  [24,48] # [8,24] # [8,24,32,48]
adp_initquery_inputemb = list(itertools.product(L_adaptive_embedding_dim,L_init_adaptive_query_dim,L_contextual_input_embedding_dim,L_input_embedding_dim))
Inpt_Adp_emb = list(itertools.product( L_input_embedding_dim,L_adaptive_embedding_dim))

dic_configs = {}
REPEAT_TRIAL  = 5 # 1 # 5
EPOCHS = 80  # 80  # 1
TORCH_COMPILE =  'compile' # 'compile'  # False
# Exp i. Fusion Strategy on STAEformer 
# --- Create configurations to evaluate ---


def add_config(dic_configs,contextual_kwargs,contextual_dataset_names,model_name,fusion_type,feature_extractor_type,add_name=''):
    dataset_names =  [target_data] +contextual_dataset_names+ ['calendar'] if contextual_kwargs is not None else  [target_data] + ['calendar']

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



if True: 
    for contextual_dataset_names in L_contextual_dataset_names: 
        # No Contextual Datasets:
        if contextual_dataset_names == []:
            contextual_kwargs,fusion_type,feature_extractor_type = None,None,None
            for horizon in horizons:
                for n_bis in range(1,REPEAT_TRIAL+1): # range(1,6):
                    dic_configs = add_config(dic_configs,contextual_kwargs,contextual_dataset_names,model_name,fusion_type,feature_extractor_type,)

        # Contextual Datasets: 
        else:
            for fusion_type, L_feature_extractor_type in dic_integration_stategies.items():
                for feature_extractor_type in L_feature_extractor_type:

                    contextual_kwargs = subway_possible_contextual_kwargs[fusion_type][feature_extractor_type]
                    for horizon in horizons:
                        for n_bis in range(1,REPEAT_TRIAL+1): # range(1,6):

                            # If Self Attention : 
                            if feature_extractor_type == 'traffic_model_backbone':
                                for Inpt_emb,adp_emb in Inpt_Adp_emb:
                                    contextual_kwargs_i = copy.deepcopy(contextual_kwargs)
                                    contextual_kwargs_i['attn_kwargs']['adaptive_embedding_dim'] = adp_emb
                                    contextual_kwargs_i['attn_kwargs']['input_embedding_dim'] = Inpt_emb
                                    add_name = f'InEmb{Inpt_emb}_adp{adp_emb}'
                                    dic_configs = add_config(dic_configs,contextual_kwargs_i,contextual_dataset_names,model_name,fusion_type,feature_extractor_type,add_name)
                            

                            # If cross attention : 
                            elif feature_extractor_type == 'adp_query_cross_attn_traffic_model_backbone':
                                for adp_emb, Q_adp, Ctx_emb, Inpt_emb in adp_initquery_inputemb:
                                    contextual_kwargs_i = copy.deepcopy(contextual_kwargs)
                                    contextual_kwargs_i['attn_kwargs']['adaptive_embedding_dim'] = adp_emb
                                    contextual_kwargs_i['attn_kwargs']['input_embedding_dim'] = Inpt_emb
                                    contextual_kwargs_i['attn_kwargs']['init_adaptive_query_dim'] = Q_adp
                                    contextual_kwargs_i['attn_kwargs']['context_input_embedding_dim'] = Ctx_emb
                                    add_name = f'InEmb{Inpt_emb}_ctxInEmb{Ctx_emb}_adp{adp_emb}_adpQ{Q_adp}'
                                    dic_configs = add_config(dic_configs,contextual_kwargs_i,contextual_dataset_names,model_name,fusion_type,feature_extractor_type,add_name)
                            
                            # If no Backbone model : 
                            else:
                                dic_configs = add_config(dic_configs,contextual_kwargs,contextual_dataset_names,model_name,fusion_type,feature_extractor_type,)
                    
print('\n------------------------------- CONFIGURATIONS ----------------------\n')
for key in list(dic_configs.keys()):
    print('   ',key)
print('\n---------------------------------------------------------------------\n')


# --- Evaluate configurations ---
loop_train_save_log(loger,dic_configs,init_save_folder = init_save_folder) 

                



