import numpy as np 
import torch
import torch.nn as nn   
from torch import Tensor
# 
# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...



from dl_models.TransformerGraphEncoder import TransformerGraphEncoder,feed_forward
from dl_models.vision_models.VariableSelectionNetwork.VariableSelectionNetwork import MultiHeadAttention


class MHA_layer(nn.Module):
    def __init__(
            self, 
            query_dim: int = 1,
            key_dim: int = 1,
            dim_model: int = 24,

            num_heads: int = 3,
            dim_feedforward: int = 32,
            output_dim: int = 1,

            dropout: float = 0.1,
            keep_topk: bool = False,
            proj_query: bool = True
            ):
        super(MHA_layer, self).__init__()

        self.mha = MultiHeadAttention(query_dim, key_dim, dim_model, num_heads, dropout, keep_topk, proj_query)
        self.feedforward = feed_forward(dim_model, dim_feedforward, output_dim)
        self.layer_norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

        #  -- Add linear proj if align is needed 
        if dim_model != output_dim:
            self.res_proj = nn.Linear(dim_model, output_dim)
        else:
            self.res_proj = None
        # -- 

    def forward(self, x_flow_station: Tensor, x_contextual: Tensor) -> Tensor:
        # ------- MHA
        projected_x_flow,context,attn_weight = self.mha(x_flow_station,x_contextual,x_contextual)
        self.attn_weight = attn_weight
        
        if self.res_proj is not None:
            residual = self.res_proj(context)
        else:
            residual = context

        # ------- Normalizing the MHA output:
        context_norm = self.layer_norm(context)
        ff_output = self.feedforward(context_norm)

        # ------- Residual connection:
        output = residual + self.dropout(ff_output)

        return projected_x_flow,output

class model(nn.Module):
    def __init__(
        self,
        query_dim: int = 1,
        key_dim: int = 1,
        dim_model: int = 24,

        num_heads: int = 3,
        dim_feedforward: int = 32,
        latent_dim: int = 1,

        dropout: float = 0.1,
        keep_topk: bool = False,
        proj_query: bool = True,
        nb_layers: int = 1
    ):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim

        if nb_layers == 1:
            self.mha_list = nn.ModuleList([MHA_layer(query_dim, key_dim, dim_model, num_heads, dim_feedforward, latent_dim, dropout, keep_topk, proj_query)])
        else:
            self.mha_list = nn.ModuleList([MHA_layer(query_dim, key_dim, dim_model, num_heads, dim_feedforward, dim_model, dropout, keep_topk, proj_query)]+
                                          [MHA_layer(dim_model, dim_model, dim_model, num_heads, dim_feedforward, dim_model, dropout, keep_topk, proj_query) for _ in range(nb_layers-2)] + 
                                          [MHA_layer(dim_model, dim_model, dim_model, num_heads, dim_feedforward, latent_dim, dropout, keep_topk, proj_query)])


        # --- Gradient we would like to Keep track on:
        tracked_grads_info = []
        for k,mha_layer in enumerate(self.mha_list):
            if hasattr(mha_layer.mha,'W_q'): 
                tracked_grads_info.append((f'layer{k}mha_W_q', mha_layer.mha.W_q))
            tracked_grads_info.extend([
                (f'layer{k}/LN_q', mha_layer.mha.layer_normq.weight),
                (f'layer{k}/LN_kv', mha_layer.mha.layer_normkv.weight),
                (f"layer{k}/mha_W_k", mha_layer.mha.W_k),
                (f"layer{k}/mha_W_v", mha_layer.mha.W_v),
                (f'layer{k}/LN_mha', mha_layer.layer_norm.weight),
                (f"layer{k}/ff_fc1", mha_layer.feedforward[0].weight),
                (f"layer{k}/ff_fc2", mha_layer.feedforward[2].weight)
            ])
        self._tracked_grads_info = tracked_grads_info
        # ----

    def forward(self, x_flow_station: Tensor, x_contextual: Tensor) -> Tensor:
        """
        inputs: 
        >>> x_flow_station [B,L,1] or [B,L,N]   (= QUERY)  
        >>> x_contextual [B,L,CP] (= KEY = VALUES)  

        Apply Spatial Attention on values 'x_contextual' and its spatial axis of dimension 'P'
        >>> x_mha : [B,L,d]  # multi-head attention   
        >>> x_fc : [B,L,z]   # Reduce to dimension z (expect z << d)

        outputs: x_fc [B,L,z*1]   or [B,L,z*N]  (where z is the output dimension of the feedforward layer)
        """
        # print('\nFoward Spatial Attention: ')
        # print('Query (x_flow_station):',x_flow_station.size())
        # print('Key/Values (x_contextual):',x_contextual.size())
        current_context = x_flow_station
        for mha_layer in self.mha_list:
            projected_x_flow, current_context = mha_layer(current_context, x_contextual)
            # print('\nProjected Flow (x_flow_station):',projected_x_flow.size())
            # print('Current Context:',current_context.size())

        return projected_x_flow,current_context