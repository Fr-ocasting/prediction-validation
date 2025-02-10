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
        L: int = 7,
        node_ids: int= 40,
        #num_layers: int = 2,
        dim_model: int = 24,
        num_heads: int = 3,
        dim_feedforward: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        #print('node_ids,num_layers,dim_model,num_heads,dim_feedforward,dropout: ',node_ids,num_layers,dim_model,num_heads,dim_feedforward,dropout)
        self.embedding = nn.Linear(L,dim_model)
        #self.spatial_attn = TransformerGraphEncoder(node_ids,num_layers,dim_model,num_heads,dim_feedforward,dropout)     
        query_dim = key_dim = dim_model
        #self.outputs = feed_forward(dim_model,dim_feedforward)
        self.mha = MultiHeadAttention(query_dim, key_dim, dim_model,num_heads,dropout)

        #self.avgpool = nn.AvgPool3d((node_ids,1,1))
        self.avgpool = nn.AvgPool2d((node_ids,1))
    def forward(self, x: Tensor) -> Tensor:
        """
        inputs: 
        >>> x [B,P,L]     
        Apply Spatial Attention on axis 'P', the spatial dim
        -> x : [B,P,L]

        outputs: x [B,1,L]   # AvgPool on spatial dim 
        """
        # [B,P,L]
        print('\nEntry SpatialAttn: ',x.size())
        print('x.size: ',x.size())
        x_emb = self.embedding(x)
        print('x.size after embedding: ',x.size())
        x_mha,attn_weight = self.mha(x_emb,x_emb,x_emb)
        print('after mha: ',x.size())

        
        print('after spatial attn: ',x.size())

        x = self.avgpool(x)
        print('after avgpool: ',x.size())

        return x