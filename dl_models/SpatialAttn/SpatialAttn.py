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



class model(nn.Module):
    def __init__(
        self,
        #node_ids: int= 40,
        #num_layers: int = 2,
        dim_model: int = 24,
        query_dim: int = 1,
        key_dim: int = 1,
        latent_dim: int = 1,
        num_heads: int = 3,
        dim_feedforward: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        #print('node_ids,num_layers,dim_model,num_heads,dim_feedforward,dropout: ',node_ids,num_layers,dim_model,num_heads,dim_feedforward,dropout)
        self.mha = MultiHeadAttention(query_dim, key_dim, dim_model,num_heads,dropout)
        self.feedforward = feed_forward(dim_model,dim_feedforward,query_dim*latent_dim)
        #self.spatial_attn = TransformerGraphEncoder(node_ids,num_layers,dim_model,num_heads,dim_feedforward,dropout)     
        #self.outputs = feed_forward(dim_model,dim_feedforward)

        #self.avgpool = nn.AvgPool3d((node_ids,1,1))
        #self.avgpool = nn.AvgPool2d((node_ids,1))
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
        #print('\nFoward Spatial Attention: ')
        #print('Query (x_flow_station):',x_flow_station.size())
        #print('Key/Values (x_contextual):',x_contextual.size())
        x_mha,attn_weight = self.mha(x_flow_station,x_contextual,x_contextual)
        #print('After MHA:',x_mha.size())
        x_fc = self.feedforward(x_mha)
        #print('After FC:',x_fc.size())

        return x_fc