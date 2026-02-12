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



from pipeline.Flex_MDI.dl_models.TransformerGraphEncoder import TransformerGraphEncoder,feed_forward
from pipeline.Flex_MDI.dl_models.vision_models.VariableSelectionNetwork.VariableSelectionNetwork import MultiHeadAttention


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
            proj_query: bool = True,

            keep_temporal_dim : bool = False
            ):
        super(MHA_layer, self).__init__()
        padding_sequence_length = not(keep_temporal_dim)
        self.keep_temporal_dim = keep_temporal_dim
        self.mha = MultiHeadAttention(query_dim, key_dim, dim_model, num_heads, dropout, keep_topk, proj_query,padding_sequence_length)
        self.layer_norm = nn.LayerNorm(dim_model)
        self.feedforward = feed_forward(dim_model, dim_feedforward, output_dim)
        self.dropout = nn.Dropout(dropout)

        #  -- Add linear proj if align is needed 
        if dim_model != output_dim:
            self.res_proj = nn.Linear(dim_model, output_dim)
        else:
            self.res_proj = None
        # -- 

    def forward(self, x_flow_station: Tensor, x_contextual: Tensor) -> Tensor:
        """
        --- Case Spatial Attention while increasing temporal dimension : ---
        >>> x_flow_station [B,L,N]   (= QUERY)
        >>> x_contextual [B,L,P] (= KEY = VALUES)

        Apply Embedding : 
        >>> x_flow_station --permute--EMB--> [B,N,d]
        >>> x_contextual --permute--EMB--> [B,P,d]

        Apply Spatial Attention on values 'x_contextual' and its spatial axis of dimension 'P'
        >>> attn_weight : [B,N,P]  #   (x_flow_station * x_contextual.T)/sqrt(d)
        >>> context : [B,N,d]  # attention output (attn_weight * x_contextual)
        >>> x_fc : [B,N,z]  # FF-Mish-FF   

        # Reduce to dimension z (expect z << d)
        ---------------------------------------------------------------------------------------------

        
        --- Case Spatial Attention while increasing channel dimension and keeping temporal dimension : ---
        >>> x_flow_station [B,L,N,C1] (= QUERY)           query_dim = C1 = 1 or z from last MHA layer
        >>> x_contextual [B,L,P,C2] (= KEY = VALUES)      key_dim = C2 = 1 as x_contextual is the same for each MHA layer

        Apply Embedding :
        >>> x_flow_station --EMB--> [B,L,N,d]             DOES IT REALLY NEED EMBEDDING AGAIN ? 
        >>> x_contextual --EMB--> [B,L,P,d]

        Apply Spatial Attention on values 'x_contextual' and its spatial axis of dimension 'P'
        >>> attn_weight : [B,L,N,P]  #   (x_flow_station * x_contextual.T)/sqrt(d)
        >>> context : [B,L,N,d]  # attention output (attn_weight * x_contextual)
        >>> x_fc : [B,L,N,z]  # FF-Mish-FF
        ---------------------------------------------------------------------------------------------
        """
        if self.keep_temporal_dim:
            if x_flow_station.dim() == 3:
                x_flow_station = x_flow_station.unsqueeze(-1)
            if x_contextual.dim() == 3:
                x_contextual = x_contextual.unsqueeze(-1)

        # ------- MHA
        projected_x_flow,context,attn_weight = self.mha(x_flow_station,x_contextual,x_contextual)
        self.attn_weight = attn_weight
        
        if self.res_proj is not None:
            residual = self.res_proj(context)
        else:
            residual = context

        # ------- Normalizing the MHA output:
        context_norm = self.layer_norm(context)
        x_fc = self.feedforward(context_norm)

        # ------- Residual connection:
        output = residual + self.dropout(x_fc)

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
        nb_layers: int = 1,
        keep_temporal_dim : bool = False
    ):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.nb_layers = nb_layers
        if nb_layers == 0:
            # No layer, just a linear projection
            self.mha_list = nn.ModuleList([ #nn.LayerNorm(latent_dim),
                                            feed_forward(key_dim, dim_feedforward, latent_dim),
                                            nn.Dropout(dropout)])
        else:
            # Apply SPatial-Attention Layers and increase channel dimension 
            if keep_temporal_dim:
                if nb_layers == 1:
                    self.mha_list = nn.ModuleList([MHA_layer(1, 1, dim_model, num_heads, dim_feedforward, latent_dim, dropout, keep_topk, proj_query,keep_temporal_dim)])
                else:
                    self.mha_list = nn.ModuleList([MHA_layer(1, 1, dim_model, num_heads, dim_feedforward, dim_model, dropout, keep_topk, proj_query,keep_temporal_dim)]+
                                                [MHA_layer(dim_model, 1, dim_model, num_heads, dim_feedforward, dim_model, dropout, keep_topk, proj_query,keep_temporal_dim) for _ in range(nb_layers-2)] + 
                                                [MHA_layer(dim_model, 1, dim_model, num_heads, dim_feedforward, latent_dim, dropout, keep_topk, proj_query,keep_temporal_dim)])
                    
            # Apply SPatial-Attention Layers and increase Temporal dimension 
            else:
                if nb_layers == 1:
                    self.mha_list = nn.ModuleList([MHA_layer(query_dim, key_dim, dim_model, num_heads, dim_feedforward, latent_dim, dropout, keep_topk, proj_query,keep_temporal_dim)])
                else:
                    self.mha_list = nn.ModuleList([MHA_layer(query_dim, key_dim, dim_model, num_heads, dim_feedforward, dim_model, dropout, keep_topk, proj_query,keep_temporal_dim)]+
                                                [MHA_layer(dim_model, dim_model, dim_model, num_heads, dim_feedforward, dim_model, dropout, keep_topk, proj_query,keep_temporal_dim) for _ in range(nb_layers-2)] + 
                                                [MHA_layer(dim_model, dim_model, dim_model, num_heads, dim_feedforward, latent_dim, dropout, keep_topk, proj_query,keep_temporal_dim)])


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

        if self.nb_layers == 0:
            current_context = x_contextual
            projected_x_flow = x_flow_station
            for layer in self.mha_list:
                current_context = layer(current_context)

        else:
            current_context = x_flow_station
            for mha_layer in self.mha_list:
                projected_x_flow, current_context = mha_layer(current_context, x_contextual)
                # print('\nProjected Flow (x_flow_station):',projected_x_flow.size())
                # print('Current Context:',current_context.size())

        return projected_x_flow,current_context