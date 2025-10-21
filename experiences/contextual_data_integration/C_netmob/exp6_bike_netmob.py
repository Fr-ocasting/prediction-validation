# Heterogenous spatial unit 


# ----------------------------------------------------------------
# Aim of the experience: 

# Evaluate performance gains when integrating NetMob contextual data (POIs)
# into the prediction of subway traffic in Lyon (Subway-Out) with
# STAEformer model.
# ----------------------------------------------------------------



# ----------------------------------------------------------------
# Limits: 

# 
# ----------------------------------------------------------------



# ----------------------------------------------------------------
# Detail of experience :

# Dataset: Subway traffic in Lyon (Subway-In, Subway-Out), Calendar data, NetMob POIs
# Model: STAEformer
# Target: Subway-In
# Contextual: NetMob POIs, Calendar
# Number of trials : 5 per configuration
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# # Configurations :
# 
# With early fusion : 
# ---
#   

# With late fusion :
# ---
#   
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
from experiences.common_parameter import possible_target_kwargs, weather_possible_contextual_kwargs, model_configurations,feature_extractor_model_configurations
from experiences.common_parameter import REPEAT_TRIAL,netmob_preprocessing_kwargs


from experiences.compilation_parameter import compilation_modification
from experiences.loop_train_save_log import loop_train_save_log

loger = LOG()

# --- Init ---  (set horizon, freq, ...)
# Set seed : NO 

init_save_folder = 'K_fold_validation/training_wo_HP_tuning/Exp6_bike_netmob'
device = torch.device('cuda:0')

freq = '15min' #'15min'  
horizons = [1,4] # [4]  #[1,4]
target_data = 'bike_out' 
contextual_dataset_names = ['netmob_POIs']

model_name = 'STAEformer'
config_backbone_model = model_configurations[model_name]
config_backbone_model['epochs'] = 150 # 150 #80
compilation_modification['torch_compile'] = 'compile' # 'compile' # 'compile'  # False 
compilation_modification['device'] = device
# REPEAT_TRIAL  = 1 

dic_configs = {}


weather_contextual_kwargs = weather_possible_contextual_kwargs['early_fusion']['repeat_t_proj']


# ------ Possible configurations :
L_input_embedding_dim = [24] # [12,24] # [24,48] # [8,24]
L_adaptive_embedding_dim = [16,32] # [16,32] # [0,16,32] 
L_init_adaptive_query_dim = [24] # [0,24] #  [0,24,48] #  [0,8,24]
L_contextual_input_embedding_dim = [8,24] # [8,12,24] #  [24,48] # [8,24] # [8,24,32,48]
L_agg_iris_target_n = [50,100]
L_NetMob_selected_apps =[['Google_Maps'], ['Web_Weather'],['Deezer'],['Instagram'],
                         ['Deezer' ,'Google_Maps'], ['Deezer' ,'Web_Weather'],['Web_Weather','Google_Maps'],
                         ['Web_Weather','Google_Maps','Deezer'],
                         ]
                         # Instagram'']
adp_initquery_inputemb = list(itertools.product(L_adaptive_embedding_dim,L_init_adaptive_query_dim,L_contextual_input_embedding_dim,L_input_embedding_dim,L_agg_iris_target_n,L_NetMob_selected_apps))
# adp_initquery_inputemb = list(itertools.product(L_adaptive_embedding_dim,L_init_adaptive_query_dim,L_input_embedding_dim))
# ------

subway_possible_contextual_kwargs = {'late_fusion':{}}


for adp_emb, init_adp_q, context_input_emb,input_emb,agg_iris_target_n,NetMob_selected_apps in adp_initquery_inputemb:
# for adp_emb, init_adp_q,input_emb in adp_initquery_inputemb:
    contextual_kwargs_i = copy.deepcopy(feature_extractor_model_configurations)
    contextual_kwargs_i.update(netmob_preprocessing_kwargs['contextual_kwargs']['netmob_POIs'])
    contextual_kwargs_i['attn_kwargs']['adaptive_embedding_dim'] = adp_emb
    contextual_kwargs_i['attn_kwargs']['init_adaptive_query_dim'] = init_adp_q
    contextual_kwargs_i['attn_kwargs']['contextual_input_embedding_dim'] = context_input_emb
    contextual_kwargs_i['attn_kwargs']['input_embedding_dim'] = input_emb
    contextual_kwargs_i['attn_kwargs']['concatenation_late'] = True
    contextual_kwargs_i['attn_kwargs']['cross_attention'] = True
    contextual_kwargs_i['NetMob_selected_apps'] = NetMob_selected_apps
    contextual_kwargs_i['backbone_model'] = True
    contextual_kwargs_i['agg_iris_target_n'] = agg_iris_target_n


    if agg_iris_target_n is not None:
        name_i = f"CrossAttnBackBone_InEmb{input_emb}_ctxInEmb{context_input_emb}_adp{adp_emb}_adpQ{init_adp_q}_aggIris{agg_iris_target_n}_{'_'.join(NetMob_selected_apps)}"
    else:
        name_i = f"CrossAttnBackBone_InEmb{input_emb}_ctxInEmb{context_input_emb}_adp{adp_emb}_adpQ{init_adp_q}_{'_'.join(NetMob_selected_apps)}"
    # name_i = f"CrossAttnBackBone_InEmb{input_emb}_adp{adp_emb}_adpQ{init_adp_q}"

    subway_possible_contextual_kwargs['late_fusion'][name_i] =  contextual_kwargs_i


print('\n------------------------------- CONFIGURATIONS ----------------------------------\n')
print(list(subway_possible_contextual_kwargs['late_fusion'].keys()))
print('\n---------------------------------------------------------------------------------\n')

# for config in subway_possible_contextual_kwargs['late_fusion'].keys():
#     print('\n---------------------------------------------------------------------------------\n')
#     print('   ',config)
#     print(subway_possible_contextual_kwargs['late_fusion'][config])


if True: 
    for fusion_type, config_contextual_kwargs in subway_possible_contextual_kwargs.items():
        for feature_extractor_type, contextual_kwargs_i in config_contextual_kwargs.items():
       
            contextual_kwargs ={'subway_out':contextual_kwargs_i,
                                'subway_in':contextual_kwargs_i,
                                'subway_in_subway_out':contextual_kwargs_i,
                                'weather':weather_contextual_kwargs,
                                'netmob_POIs':contextual_kwargs_i
                                }
            
            if 'weather' not in contextual_dataset_names:
                contextual_kwargs.pop('weather',None)  
            if 'subway_in' not in contextual_dataset_names:
                contextual_kwargs.pop('subway_in',None)  
            if 'subway_out' not in contextual_dataset_names:
                contextual_kwargs.pop('subway_out',None)
            if 'subway_in_subway_out' not in contextual_dataset_names:
                contextual_kwargs.pop('subway_in_subway_out',None) 
            if 'netmob_POIs' not in contextual_dataset_names:
                contextual_kwargs.pop('netmob_POIs',None)


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
                    
                    if 'denoiser_kwargs' in netmob_preprocessing_kwargs.keys():
                        config_i.update({'denoising_names':['netmob_POIs'],
                                        'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                                        'denoiser_kwargs': netmob_preprocessing_kwargs['denoiser_kwargs']}
                                        )
                        
                    config_i.update(config_backbone_model)
                    config_i.update(compilation_modification)

                    dic_configs[name_i] = config_i



if True: 
    contextual_kwargs ={}
    for horizon in horizons:
        for n_bis in range(1,REPEAT_TRIAL+1): # range(1,6):
            dataset_names =  [target_data] + ['calendar']
            name_i = f"{model_name}_{'_'.join(dataset_names)}"
            name_i_end = f"_e{config_backbone_model['epochs']}_h{horizon}_bis{n_bis}"
            name_i = f"{name_i}_{name_i_end}"

            config_i =  {'target_data': target_data,
                        'dataset_names': dataset_names,
                        'model_name': model_name,
                        'dataset_for_coverage': [target_data,'netmob_POIs'],
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

                


