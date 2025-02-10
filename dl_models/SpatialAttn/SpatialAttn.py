import numpy as np 
import torch
import torch.nn as nn

# 
# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...


from torch.autograd import Variable
from torch import Tensor
import math
import torch.nn.functional as F

from dl_models.TransformerGraphEncoder import TransformerGraphEncoder


class model(nn.Module):
    def __init__(
        self,
        node_ids: int = 22,
        num_layers: int = 6,
        dim_model: int = 128,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spatial_attn = TransformerGraphEncoder(node_ids,num_layers,dim_model,num_heads,dim_feedforward,dropout)      
        self.avgpool = nn.AvgPool3d((node_ids,1,1))
    def forward(self, x: Tensor) -> Tensor:
        """
        inputs: 
        >>> x [B,P,1,L]     
        Apply Spatial Attention on axis 'P', the spatial dim
        -> x : [B,P,1,L]

        outputs: x [B,1,1,L]   # AvgPool on spatial dim 
        """
        # [B,P,L] -> [B,1,P,L]
        if x.dim()==3:
            x.unsqueeze(1)

        # [B,1,P,L] -> [B,P,1,L]
        x = x.permute(0,2,1,3)

        print(x.size())
        x = self.spatial_attn(x)
        print('after spatial attn: ',x.size())

        x = self.avgpool(x)
        print('after avgpool: ',x.size())

        return x