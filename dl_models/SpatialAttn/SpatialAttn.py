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
        query_dim: int = 7,
        key_dim: int = 7,
        num_heads: int = 3,
        dim_feedforward: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        #print('node_ids,num_layers,dim_model,num_heads,dim_feedforward,dropout: ',node_ids,num_layers,dim_model,num_heads,dim_feedforward,dropout)
        self.feedforward = feed_forward(dim_model,dim_feedforward,query_dim)
        #self.spatial_attn = TransformerGraphEncoder(node_ids,num_layers,dim_model,num_heads,dim_feedforward,dropout)     
        #self.outputs = feed_forward(dim_model,dim_feedforward)
        self.mha = MultiHeadAttention(query_dim, key_dim, dim_model,num_heads,dropout)

        #self.avgpool = nn.AvgPool3d((node_ids,1,1))
        #self.avgpool = nn.AvgPool2d((node_ids,1))
    def forward(self, x_flow_station: Tensor, x_contextual: Tensor) -> Tensor:
        """
        inputs: 
        >>> x_flow_station [B,L]   (= Query)  
        >>> x_contextual [B,P,L]   (= Key = Values)  

        Apply Spatial Attention on values 'x_contextual' and is spatial axis of dimension 'P'
        >>> x_mha : [B,d]  # multi-head attention   (B,N,d works also)
        >>> x_fc : [B,P,L]   # go back to dimension L 
        >>> x_agg : [B,1,L]  # Average Pooling on spatial dim 


        outputs: x_contextual [B,1,L]   # AvgPool on spatial dim 
        """
        #print('\nEntry SpatialAttn: ')
        #print('x_flow_station.size(): ',x_flow_station.size(),'x_contextual: ',x_contextual.size())
        x_mha,attn_weight = self.mha(x_flow_station,x_contextual,x_contextual)
        #print('after mha: ',x_mha.size())
        x_fc = self.feedforward(x_mha)
        #print('after feedforward: ',x_fc.size())
        #if x_fc.dim()==3:
        #    x_fc = self.avgpool(x_fc)

        return x_fc