
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Optional,List
import torch
from torch import Tensor
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from dl_models.TransformerGraphEncoder import TransformerGraphEncoder

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):   
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, num_nodes = x.shape

            padding =  torch.zeros([batch_size, self.c_out - self.c_in, timestep, num_nodes]).to(x)
            x = torch.cat([x,padding], dim=1)
        else:
            x = x
        
        return x
"""
class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        if enable_padding == True:
            self.ca1d_padding = (kernel_size - 1) * dilation
        else:
            self.ca1d_padding = 0
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=self.ca1d_padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.ca1d_padding != 0:
            return result[: , : , : -self.ca1d_padding]
        
        return result
"""
class CausalConv2d(nn.Module): # nn.Conv2d
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        super().__init__()
        self.kernel_size = nn.modules.utils._pair(kernel_size)
        self.stride = nn.modules.utils._pair(stride)
        self.dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.ca2d_padding = [int((self.kernel_size[i] - 1) * self.dilation[i]) for i in range(len(self.kernel_size))]
        else:
            self.ca2d_padding = 0
        self.left_padding = nn.modules.utils._pair(self.ca2d_padding)
        ## REMOVED TO ALLOW TORCH DYNAMO COMPILE 
        #  #super(CausalConv2d, self).__init__(in_channels, out_channels, self.kernel_size, stride=self.stride, padding=0, dilation=self.dilation, groups=groups, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv2d = nn.Conv2d(in_channels,out_channels, kernel_size=self.kernel_size,  stride=self.stride, padding=0, dilation=self.dilation,groups=groups,bias=bias)
    def forward(self, input):
        if self.ca2d_padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
            #print(f'x after padding: {input.size()}, Args F.pad: ({self.left_padding[1]}, 0, {self.left_padding[0]}, 0)')
        #print('kernel size: ',self.kernel_size)
        # REMOVED TO ALLOW TORCH DYNAMO COMPILE  
        # #result = super(CausalConv2d, self).forward(input)
        result = self.conv2d(input)
        #print(f'x after causal conv2D: {result.size()}')    

        return result

class TemporalConvLayer(nn.Module):

    # Temporal Convolution Layer (GLU)
    #
    #        |--------------------------------| * residual connection *
    #        |                                |
    #        |    |--->--- casualconv2d ----- + -------|       
    # -------|----|                                   ⊙ ------>
    #             |--->--- casualconv2d --- sigmoid ---|                               
    #
    
    #param x: tensor, [bs, c_in, ts, num_nodes]
    def __init__(self, Kt, c_in, c_out, num_nodes, act_func,enable_padding):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.num_nodes = num_nodes
        self.enable_padding = enable_padding
        self.align = Align(c_in, c_out)
        if act_func == 'glu' or act_func == 'gtu':
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(Kt, 1), enable_padding=enable_padding, dilation=1)
        else:
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=c_out, kernel_size=(Kt, 1), enable_padding=enable_padding, dilation=1)

        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.act_func = act_func

    def forward(self, x):   
        '''
        x : [B,C,L,N]
        kernel-size : (Kt,1)

        >>>  Apply 2D conv through the spatio-temporal space [L,N], with a temporal window Kt > 1 and spatial window = 1 

        outputs:
        ---------
        x_out : [B,C',L-(Kt-1),N]
        
        '''
        # Enable padding permet d'avoir +2 après la causal_conv, mais il faut aussi ajouter +2 à x_in
        # =========
        # MODIFICATION EFFECTUEE ICI AVEC LE SELF.ENABLE_PADDING QUI N EXISTAIT PAS 
        # =========
        #print('Entry TemporalConvLayer')
        #print('x: ',x.size())

        # Align Residual : 
        x_in = self.align(x)

        if not(self.enable_padding):
            x_in = x_in[:, :, self.Kt - 1:, :]  

        #print('x after align: ',x_in.size()) 
        x_causal_conv = self.causal_conv(x)
        #print('x after causal conv: ',x_causal_conv.size())

        if self.act_func == 'glu' or self.act_func == 'gtu':
            x_p = x_causal_conv[:, : self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out:, :, :]

            if self.act_func == 'glu':
                # Explanation of Gated Linear Units (GLU):
                # The concept of GLU was first introduced in the paper 
                # "Language Modeling with Gated Convolutional Networks". 
                # URL: https://arxiv.org/abs/1612.08083
                # In the GLU operation, the input tensor X is divided into two tensors, X_a and X_b, 
                # along a specific dimension.
                # In PyTorch, GLU is computed as the element-wise multiplication of X_a and sigmoid(X_b).
                # More information can be found here: https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.glu
                # The provided code snippet, (x_p + x_in) ⊙ sigmoid(x_q), is an example of GLU operation. 
                x = torch.mul((x_p + x_in), torch.sigmoid(x_q))

            else:
                # tanh(x_p + x_in) ⊙ sigmoid(x_q)
                x = torch.mul(torch.tanh(x_p + x_in), torch.sigmoid(x_q))

        elif self.act_func == 'relu':
            x = self.relu(x_causal_conv + x_in)

        elif self.act_func == 'silu':
            x = self.silu(x_causal_conv + x_in)
        
        else:
            raise NotImplementedError(f'ERROR: The activation function {self.act_func} is not implemented.')
        
        return x

class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, gso, bias,g_constructor,graph_conv_type,Ks=None):
        super(GraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gso = gso
        self.idx = torch.arange(gso.shape[0]).to(gso) #self.num_nodes  #self.num_nodes 
        self.g_constructor = g_constructor
        #print('number of parameters in g_constructor: {}'.format(sum([p.numel() for p in g_constructor.parameters()])))
        self.Ks = Ks

        self.graph_conv_type = graph_conv_type
        if graph_conv_type == 'cheb_graph_conv':
            self.weight = nn.Parameter(torch.cuda.FloatTensor(Ks,c_in, c_out)) if torch.cuda.is_available() else nn.Parameter(torch.FloatTensor(Ks,c_in, c_out))
        elif graph_conv_type == 'graph_conv':
            self.weight = nn.Parameter(torch.cuda.FloatTensor(c_in, c_out)) if torch.cuda.is_available() else nn.Parameter(torch.FloatTensor(c_in, c_out))
        else:
            raise NotImplementedError(f"graph_conv_type {self.graph_conv_type} has not been implemented")

        if bias:
            self.bias = nn.Parameter(torch.cuda.FloatTensor(c_out)) if torch.cuda.is_available() else nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def compute_graph_conv(self,x):
        first_mul = torch.einsum('hi,btij->bthj', self.gso, x)
        graph_conv = torch.einsum('bthi,ij->bthj', first_mul, self.weight)
        return graph_conv

    def compute_cheb_graph_conv(self,x):
        x_list = []
        if self.Ks - 1 < 0:
            raise ValueError(f'ERROR: the graph convolution kernel size Ks has to be a strict positive integer, but received {self.Ks}.')  
        elif self.Ks - 1 == 0:
            x_0 = x
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(torch.einsum('hi,btij->bthj', 2 * self.gso, x_list[k - 1]) - x_list[k - 2])
        
        x = torch.stack(x_list, dim=2)
        graph_conv = torch.einsum('btkhi,kij->bthj', x, self.weight)
        return graph_conv
    
    def add_bias(self,graph_conv):
        if self.bias is not None:
            graph_conv = torch.add(graph_conv, self.bias)
        return graph_conv

    def forward(self, x):
        # If learnable adjacency matrix: 
        if self.g_constructor is not None:    
            self.gso = self.g_constructor(self.idx.long())
        x = x.permute(0, 2, 3, 1)

        if self.graph_conv_type == 'cheb_graph_conv':
            graph_conv = self.compute_cheb_graph_conv(x)
        elif self.graph_conv_type == 'graph_conv':
            graph_conv = self.compute_graph_conv(x)
        else:
            raise NotImplementedError(f"graph_conv_type {self.graph_conv_type} has not been implemented")
        
        graph_conv = self.add_bias(graph_conv)
            
        return graph_conv


class GraphConvLayer(nn.Module):
    def __init__(self, graph_conv_type, c_in, c_out, Ks, gso, bias,g_constructor):
        super(GraphConvLayer, self).__init__()
        self.graph_conv_type = graph_conv_type
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.Ks = Ks
        self.gso = gso
        """
        if self.graph_conv_type == 'cheb_graph_conv':
            self.cheb_graph_conv = ChebGraphConv(c_out, c_out, Ks, gso, bias,g_constructor)
        elif self.graph_conv_type == 'graph_conv':
            self.graph_conv = GraphConv(c_out, c_out, gso, bias,g_constructor)"
        """
        self.graph_conv = GraphConv(c_out, c_out, gso, bias,g_constructor,graph_conv_type,Ks)
    def forward(self, x):
        x_gc_in = self.align(x)
        x_gc = self.graph_conv(x_gc_in)

        """
        if self.graph_conv_type == 'cheb_graph_conv':
            x_gc = self.cheb_graph_conv(x_gc_in)
        elif self.graph_conv_type == 'graph_conv':
            x_gc = self.graph_conv(x_gc_in)
        """
        x_gc = x_gc.permute(0, 3, 1, 2)
        x_gc_out = torch.add(x_gc, x_gc_in)

        return x_gc_out

class STConvBlock(nn.Module):
    # STConv Block contains 'TGTND' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv or GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    def __init__(self, Kt, Ks, num_nodes, last_block_channel, channels, act_func, graph_conv_type, gso, bias, dropout,enable_padding = False,g_constructor = None):
        super(STConvBlock, self).__init__()
        # ...
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], num_nodes, act_func,enable_padding)
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias, g_constructor)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], num_nodes, act_func,enable_padding)
        self.tc2_ln = nn.LayerNorm([num_nodes, channels[2]])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        '''Inputs: x: [B,C,L,N] 

        # --------------------------------------------------------- *(stblock_num)
        # Temporal Conv1: x [B,C,L,N]  --> [B,C,L,N]
        # Spatial Conv  :
        # Temporal Conv2:
        # ---------------------------------------------------------


        Outputs:
        --------
        x_out :  [B, C_out, L-4*nb_blocks, N]
        
        '''
        #print('\nShape avant de rentrer dans tmp_conv1: ',x.size())
        x = self.tmp_conv1(x)
        #print('\nShape après tmp_conv1: ',x.size())
        x = self.graph_conv(x)
        #print('\nShape après graph conv: ',x.size())
        x = self.relu(x)
        x = self.tmp_conv2(x)
        #print('\nShape après tmp_conv2 conv: ',x.size())
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)
        #print('\nShape en sortie du STConvBlock: ',x.size(),'\n')

        return x

class OutputBlock(nn.Module):
    # Output block contains 'TNFF' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, Ko, last_block_channel, channels, end_channel, num_nodes, act_func, bias, dropout,
                 concatenation_late,extracted_feature_dim,
                 TE_concatenation_late,embedding_dim,temporal_graph_transformer_encoder,
                 TGE_num_layers=None, TGE_num_heads=None,TGE_FC_hdim=None,
                 ):
        super(OutputBlock, self).__init__()
        self.temporal_graph_transformer_encoder = temporal_graph_transformer_encoder
        self.temporal_agg = nn.Identity() # initialize temporal agg for torch.jit.script
        if temporal_graph_transformer_encoder:
            if False:
                self.temporal_agg = TransformerGraphEncoder(node_ids = num_nodes,
                                                            num_layers = TGE_num_layers,
                                                            num_heads = TGE_num_heads,
                                                            dim_model = last_block_channel,
                                                            dim_feedforward = TGE_FC_hdim,
                                                            dropout =dropout
                                                            )
            self.temporal_agg = TransformerGraphEncoder(node_ids = num_nodes,
                                                        num_layers = TGE_num_layers,
                                                        num_heads = TGE_num_heads,
                                                        dim_model = Ko,
                                                        dim_feedforward = TGE_FC_hdim,
                                                        dropout =dropout
                                                        )
            
        self.temporal_conv_out = TemporalConvLayer(Ko, last_block_channel, channels[0], num_nodes, act_func,enable_padding = False)
                                              

        # Design Input Dimension according to contextual data integration or not: 
        in_channel_fc1 = channels[0]  #blocks[-2][0]
        if concatenation_late:
            in_channel_fc1 = in_channel_fc1 + extracted_feature_dim
        if TE_concatenation_late:
            in_channel_fc1 = in_channel_fc1 +embedding_dim

        self.concatenation_late = concatenation_late
        self.TE_concatenation_late = TE_concatenation_late
        # ...
        if False:
            if temporal_graph_transformer_encoder:
                # FC1: [last_block_channel,channels[1]]. Here 'channels[0] never used. Cause we don't change the C dim with TransformerGraphEncoder. 
                self.fc1 = nn.Linear(in_features=last_block_channel, out_features=channels[1], bias=bias)
        else:
            self.fc1 = nn.Linear(in_features=in_channel_fc1, out_features=channels[1], bias=bias)
        self.fc2 = nn.Linear(in_features=channels[1], out_features=end_channel, bias=bias)
        self.tc1_ln = nn.LayerNorm([num_nodes, channels[0]])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward_temporal_agg(self,x):
        '''
        Reduce the temporal dimension to 1. 

        inputs: x [B,C,L,N]
        >>> Permute for Temporal MHA: x [B,C,L,N] -> [B,N,L,C]
        >>> after TemporalConvLayer or TemporalGraphTransformerEncoder: x [B,C,L,N] -- >[B,C',N,1]
        >>> after permute: [B,C',N,1] --> [B,1,N,C']

        >>> If same than Rim paper: 
            outputs MHA:[B,C,N,L]
            after temporal agg: [B,1,N,C]

        >>> If expected Temporal Graph Encoding: 
            outputs MHA:[B,L,N,C]
            after temporal agg: [B,1,N,C]
        '''
        #print('x before temporal MHA: ',x.size())
        if not(x.numel() == 0):
            if self.temporal_graph_transformer_encoder:
                ''' Temporal Graph Encoder where we project axis L into latent space: '''
                # [B,C,L,N] --permute--> [B,C,N,L] 
                x = x.permute(0,1,3,2)
                # [B,C,N,L] --Temporal PointWise Convolution--> [B,C,N,L'] --ScaledDotProduct--> [B,C,N,L']
                x = self.temporal_agg(x) 
                #print('x after temporal MHA: ',x.size())

                #[B,C,N,L'] -> [B,C,L',N]
                x = x.permute(0,1,3,2)

                ''' Temporal Graph Encoder as implemented in Rim Paper '''
                if False:
                    # [B,C,L,N] --permute--> [B,L,N,C] 
                    x = x.permute(0,2,3,1)
                    # [B,L,N,C]--Temporal PointWise Convolution--> [B,L',N,C]-->permute(0,3,2,1)-->[B,C,N,L'] --ScaledDotProduct--> [B,C,N,L']
                    x = self.temporal_agg(x) 
                    #print('x after temporal agg: ',x.size())

                    #[B,C,N,L'] -> [B,C',L,N]
                    x = x.permute(0,1,3,2)


            # [B,C,L,N]  -> [B,C,1,N]
            #print('x before temporal conv: ',x.size())
            x = self.temporal_conv_out(x)
            #print('x after temporal conv: ',x.size())

            # Permute [B,C,1,N]  -> [B,1,N,C]
            x = self.tc1_ln(x.permute(0, 2, 3, 1)) 
    
        #print('x after norm and permute: ',x.size())
        return x


    def forward(self,x: Tensor,
                x_vision: Optional[Tensor] = None, 
                x_calendar: Optional[Tensor] = None) -> Tensor:
        # print("\nEntry Output Block:")
        # print('x.size(): ',x.size())   #->  [B,C,N,1]
        x = self.forward_temporal_agg(x)

        # print('x.size after temporal conv + permute: ',x.size())

        ## Concatenate Late if exists:
        cat_list: List[Tensor] = []
        if self.concatenation_late and x_vision is not None:
            #assert x_vision is not None
            # Concat [B,1,N,Z] + [B,1,N,L'] -> [B,1,N,Z+L']
            # print('x_vision.size(): ',x_vision.size())
            x_vision = x_vision.unsqueeze(1)  # [B,N,Z] -> [B,1,N,Z] 
            cat_list.append(x_vision) 
        if self.TE_concatenation_late and x_calendar is not None:
            # Reduce channel dim of calendar to 1 (cause correspond to repeated channel and x channel has been reduce to 1)
            # print('x_calendar.size(): ',x_calendar.size())
            x_calendar = x_calendar[:,0:1,:,:]  # [B,C,N,L_calendar] -> [B,1,N,L_calendar]
            # print('keep only one dimension (all identical): x_calendar.size(): ',x_calendar.size())
            # Concat [B,1,N,Z] + [B,1,N,L_calendar]-> [B,1,N,Z+L_calendar]
            cat_list.append(x_calendar)
        if len(cat_list) > 0:
            x = torch.cat([x]+cat_list,dim=-1) 
        
        ## ...

        # print('x.size after concatenation late if exists: ',x.size())
        # print("\nforward output module:")
        # print('fc1: ',self.fc1)
        x = self.fc1(x)
        #print('x after fc1: ',x.size())
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        #print('x.size after fc2: ',x.size())
        x = x.permute(0, 3, 1, 2)
        #print('output (after permute): ',x.size())

        return x