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

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..','..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from experiences.common_parameter import subway_possible_contextual_kwargs

dic_integration_stategies = {
    'early_fusion':['s_proj_t_proj',
                    'shared_embedding', 
                    'independant_embedding',
                    'traffic_model_backbone',
                    # 'adp_query_cross_attn_traffic_model_backbone',
                    ],
    'late_fusion':[
                    # 'adp_query_cross_attn_traffic_model_backbone',
                    'traffic_model_backbone',
                    'simple_embedding'
                    ],
                 }


possible_contextual_kwargs = {'late_fusion':{},
                              'early_fusion':{}}

def get_possible_contextual_kwarg(add_name_save):
    for fusion_type, L_feature_extractor_type in dic_integration_stategies.items():
        for feature_extractor_type in L_feature_extractor_type:
            potential_config =  copy.deepcopy(subway_possible_contextual_kwargs[fusion_type][feature_extractor_type])

            if feature_extractor_type == 'traffic_model_backbone':
                potential_config['input_embedding_dim'] = 24
                potential_config['adaptive_embedding_dim'] = 16
                end_trial_id=f"BackBone_InEmb{potential_config['input_embedding_dim']}_adp{potential_config['adaptive_embedding_dim']}"
            else:
                end_trial_id= f'{feature_extractor_type}'
            
            
            trial_id = f'{end_trial_id}{add_name_save}'

            possible_contextual_kwargs[fusion_type][trial_id] = potential_config

    return possible_contextual_kwargs

