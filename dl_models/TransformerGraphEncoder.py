import torch
import torch.nn as nn   
from torch.autograd import Variable
from torch import Tensor
import math
import torch.nn.functional as F

def feed_forward(dim_input: int = 128, dim_feedforward: int = 512,dim_output: int=None) -> nn.Module:
    if dim_output is None:
        dim_output = dim_input
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward,dtype=torch.float),
        nn.Mish(),
        nn.Linear(dim_feedforward, dim_output,dtype=torch.float),
    )
    
class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension,dtype=torch.float)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        x=self.dropout(self.sublayer(*tensors))
        x=tensors[0] + x
        x=self.norm(x)
        return x

class AttentionHead(nn.Module):
    def __init__(self, dim_model: int, dim_v: int, dim_k: int,kernel_size: int = 1 , stride :int =1):
        super().__init__()
        self.d_k=dim_k
        self.d_v=dim_v
        self.q_conv=nn.Conv2d(
                dim_model,
                dim_k,
                kernel_size=(kernel_size, 1),
                padding=(int((kernel_size - 1) / 2), 0),
                stride=(stride, 1),dtype=torch.float)
        self.k_conv=nn.Conv2d(
                dim_model,
                dim_k,
                kernel_size=(kernel_size, 1),
                padding=(int((kernel_size - 1) / 2), 0),
                stride=(stride, 1),dtype=torch.float)
        self.v_conv=nn.Conv2d(
                dim_model,
                dim_v,
                kernel_size=(kernel_size, 1),
                padding=(int((kernel_size - 1) / 2), 0),
                stride=(stride, 1),dtype=torch.float)
        
        self.softmax = nn.Softmax(-1)
    def attention(self,Q,K,V):
        sqrt_dk=torch.sqrt(torch.tensor(self.d_k))
        attention_weights=self.softmax((Q @ K.transpose(-2,-1))/sqrt_dk) # F.softmax((Q @ K.transpose(-2,-1))/sqrt_dk))
        attention_vectors=attention_weights @ V

        return attention_vectors            
    def forward(self, x: Tensor) -> Tensor:
        '''
        x : [B,L,N,C]   
        PE : [B,L,N,C]

        >>> x = x+ PE   

        ### Spatio-Temporal PointWise Convolution.  C' = C//n_head
        >>> permute(0,3,2,1) [B,L,N,C]--> [B,C,N,L]  
        >>> q_conv(): [B,C,N,L] --> [B,C',N,L]              
        >>> permute(0,3,2,1): [B,C',N,L]--> [B,L,N,C']

        ### Self Attention :
        >>> scaled dot product: [B,L,N,C']--> [B,L,N,C']
        '''
        #print('\nHead')
        #print('Input q,k,v before conv and permute: ',x.size())
        batch_size = x.size(0)
        seq_length = x.size(1)
        graph_size=x.size(2)

        x=x.permute(0,3,2,1)
        #print('Input q,k,v after permute: ',x.size())
        # x=x.transpose(1,2)
        #Q, K, V=torch.split(self.qkv_conv(x), [self.d_k , self.d_k, self.d_v],
        #                            dim=1)
        Q=self.q_conv(x).permute(0,3,2,1)
        K=self.k_conv(x).permute(0,3,2,1)
        V=self.v_conv(x).permute(0,3,2,1)

        #print('Input q,k,v after conv and permuted again: ',Q.size(),K.size(),V.size())

        x=self.attention(Q,K,V).transpose(1,2).contiguous().view(batch_size,seq_length,graph_size, self.d_k)
        
        #print('x after attention: ',x.size())
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model: int, num_heads: int,dim_k,dim_q,dim_v):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_model, dim_v, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_model,dtype=torch.float)

    def forward(self, x) -> Tensor:
        outs=[]
        #print('Input MHA: ',x.size())
        for h in self.heads:
            outs.append(h(x))
        outs=torch.cat(outs, dim=-1)

        #print('Outs n_head concatenated: ',outs.size())
        outs=self.linear(
            outs
        )

        return outs

class TransformerGraphEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 64,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        ):
        super().__init__()
        dim_v=dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(
            MultiHeadAttention(dim_model,num_heads,dim_v,dim_q,dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(dim_model,dtype=torch.float)
    def forward(self, src: Tensor) -> Tensor:
        #print('\nStart Transformer Graph Encoder Layer: ')
        #print('src before attention: ',src.size())
        src = self.attention(self.norm(src))
        #print('src after attention: ',src.size())
        src = self.feed_forward(src)
        #print('src after 2FC: ',src.size())
        return self.feed_forward(src)

class PositionalEncoder(nn.Module):
    def __init__(self, dim_model,node_ids = 22, max_seq_len = 256):
        '''
        dim_model: nb channels of the inputs of the TransformerGraphEncoder (i.e output of the traffic core-model)
                    but also the total dimension of embedding for the TransformerGraphEncoder
        node_ids: nb of graph Nodes ???
        '''
        
        super().__init__()
        self.dim_model = dim_model
        
        # create constant 'pe' matrix with values dependant on z
        # pos and i
        pe = torch.zeros(max_seq_len,node_ids , dim_model)  # [B,L,N,C]   x + PE = x'   [ B,L,N,C]
        for pos in range(max_seq_len):
          for node_id in range(0,node_ids) :
            for i in range(0, dim_model, 2):
                pe[pos, node_id, i] = \
                math.sin(pos / (10000 ** ((2 * i)/dim_model)))
                if i + 1 < dim_model:  # In case dim_model is not odd: 
                    pe[pos, node_id, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1))/dim_model)))
                
        pe = pe.unsqueeze(0)
        #self.learnable_pe=nn.Linear(d_model, d_model,dtype=torch.float)
        self.norm=nn.LayerNorm(dim_model,dtype=torch.float)
        self.register_buffer('pe', pe)

    
    def forward(self, x):
        # make embeddings relatively larger
        # x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        #print('\nStart positional Encoder: ')
        #print('x.size: ', x.size())
        #print('self.pe[:,:seq_len,:,:]: ', self.pe[:,:seq_len,:,:].size())

        if torch.cuda.is_available():
            x = self.norm(x + Variable(self.pe[:,:seq_len,:,:],requires_grad=False).cuda(x.device))
        else:
            #print('x.size(): ',x.size())
            #print('self.pe.size(): ',self.pe.size())
            x = self.norm(x + Variable(self.pe[:,:seq_len,:,:],requires_grad=False))
        
        return x

class TransformerGraphEncoder(nn.Module):
    def __init__(
        self,
        node_ids: int = 22,
        num_layers: int = 6,
        dim_model: int = 128,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        bool_positional_encoder: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerGraphEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)      
            for _ in range(num_layers)
            ]
        )
        self.bool_positional_encoder = bool_positional_encoder
        if bool_positional_encoder:
            self.positional_encoder=PositionalEncoder(dim_model,node_ids)
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Temporal Attention on axis 'L'.
        inputs: x [B,L,N,C]  

        // x --PointWise Temporal Convolution-->[B,L',N,C]-->permute(0,3,2,1)-->[B,C,N,L']-->MHA-->[B,C,N,L']
        // >>> q_conv=nn.Conv2d(dim_model,dim_model//n_heads,kernel_size=(1, 1)))
        // >>>  Q=self.q_conv(x).permute(0,3,2,1)
        // >>> Attn_coeff: [B,C,N,N]

        """

        #print('\nentry of the Temporal MHA: ',x.size())
        if self.bool_positional_encoder:
            x += self.positional_encoder(x)
        #print('x after add PE + norm: ',x.size())
        for layer in self.layers:
            x = layer(x)
            #print('x.size after layer: ', x.size())

        #print('output from the temporal MHA (before output FC layer): ',x.size())

        return x