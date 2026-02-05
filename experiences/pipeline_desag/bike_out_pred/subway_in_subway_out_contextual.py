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

from experiences.common_parameter import feature_extractor_model_configurations


# ------ Possible configurations :
L_input_embedding_dim = [24] # [12,24] # [24,48] # [8,24]
L_adaptive_embedding_dim = [16] # [16,32] # [0,16,32] 
L_init_adaptive_query_dim = [24,0] #[0,24] #  [0,24,48] #  [0,8,24]
L_contextual_input_embedding_dim = [8] # [8,12,24] #  [24,48] # [8,24] # [8,24,32,48]
adp_initquery_inputemb = list(itertools.product(L_adaptive_embedding_dim,L_init_adaptive_query_dim,L_contextual_input_embedding_dim,L_input_embedding_dim))
# ------


def get_possible_contextual_kwarg(add_name_save):

    possible_contextual_kwargs = {'early_fusion':{},
                                   'late_fusion':{}}
    
    for adp_emb, init_adp_q, context_input_emb,input_emb in adp_initquery_inputemb:
    # for adp_emb, init_adp_q,input_emb in adp_initquery_inputemb:
        contextual_kwargs_i = copy.deepcopy(feature_extractor_model_configurations)
        contextual_kwargs_i['attn_kwargs']['adaptive_embedding_dim'] = adp_emb
        contextual_kwargs_i['attn_kwargs']['init_adaptive_query_dim'] = init_adp_q
        contextual_kwargs_i['attn_kwargs']['contextual_input_embedding_dim'] = context_input_emb
        contextual_kwargs_i['attn_kwargs']['input_embedding_dim'] = input_emb
        contextual_kwargs_i['attn_kwargs']['cross_attention'] = True
        contextual_kwargs_i['backbone_model'] = True


        name_i = f"CrossAttnBackBone_InEmb{input_emb}_ctxInEmb{context_input_emb}_adp{adp_emb}_adpQ{init_adp_q}"

        contextual_kwargs_i['attn_kwargs']['concatenation_late'] = True
        possible_contextual_kwargs['late_fusion'][name_i] =  copy.deepcopy(contextual_kwargs_i)

        # contextual_kwargs_i['attn_kwargs']['concatenation_late'] = False
        # possible_contextual_kwargs['early_fusion'][name_i] =  copy.deepcopy(contextual_kwargs_i)
    return possible_contextual_kwargs

