import torch
import torch.nn as nn

# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal import:
import dl_models.STGCN.STGCN_layer as layers
from dl_models.MTGNN.MTGNN_layer import graph_constructor
# ============================================================
# Inspired by  https://github.com/hazdzz/STGCN/tree/main
# ============================================================

#  -- STGCN Layer x L -- > output Module -- > Prediction 
#
# output Module =  Convolution (Pooling sur un axe) -- FC1 -- ReLU -- FC2 

class STGCN(nn.Module):
    # STGCN contains 'TGTND TGTND TNFF' structure

    # GraphConv is the graph convolution from GCN.
    # GraphConv is not the first-order ChebConv, because the renormalization trick is adopted.

    # ChebGraphConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind as a graph filter.
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, args, gso, blocks,Ko):
        super(STGCN, self).__init__()

        self.out_dim = blocks[-1][-1]
        modules = []
        self.init_learnable_adjacency_matrix(args.learnable_adj_matrix,
                                             args.n_vertex,
                                             k=args.learnable_adj_top_k if getattr(args,'learnable_adj_matrix') else None,
                                             node_embedding_dim=args.learnable_adj_embd_dim if getattr(args,'learnable_adj_matrix') else None,
                                             device = args.device,
                                             alpha=3)



        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, args.n_vertex, blocks[l][-1], blocks[l+1], args.act_func, args.graph_conv_type, gso, args.enable_bias, args.dropout,args.enable_padding,self.g_constructor))
        self.st_blocks = nn.Sequential(*modules)

        self.vision_concatenation_late = args.args_vision.concatenation_late if hasattr(args.args_vision,'concatenation_late') else False
        self.TE_concatenation_late = args.args_embedding.concatenation_late if hasattr(args.args_embedding,'concatenation_late') else False 

        self.Ko = Ko
        self.n_vertex = args.n_vertex
        if hasattr(args.args_vision,'out_dim'):
            extracted_feature_dim = args.args_vision.out_dim 
        else:
            extracted_feature_dim = None

        if hasattr(args.args_embedding,'embedding_dim'):
            embedding_dim = args.args_embedding.embedding_dim 
        else:
            embedding_dim = None


        in_feature_fc1 = blocks[-3][-1] 

        if self.Ko > 0:
            #print('blocks: ',blocks)
            #print('in_feature_fc1: ',in_feature_fc1)
            self.output = layers.OutputBlock(self.Ko, in_feature_fc1, blocks[-2], blocks[-1][0], args.n_vertex, args.act_func, args.enable_bias, args.dropout,
                                             self.vision_concatenation_late,extracted_feature_dim,
                                             self.TE_concatenation_late,embedding_dim,args.temporal_graph_transformer_encoder,
                                             TGE_num_layers=args.TGE_num_layers, TGE_num_heads=args.TGE_num_heads, TGE_FC_hdim=args.TGE_FC_hdim
                                             )
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=in_feature_fc1, out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()


            #self.leaky_relu = nn.LeakyReLU()
            #self.silu = nn.SiLU()
            #self.dropout = nn.Dropout(p=args.dropout)

    def init_learnable_adjacency_matrix(self,bool_learnable_adj,n_vertex,k,node_embedding_dim,device,alpha):
        if bool_learnable_adj:
            self.g_constructor = graph_constructor(n_vertex, k, node_embedding_dim, device=device, alpha=alpha, static_feat=None).to(device)     
        else:
            self.g_constructor = None


    def forward(self, x,x_vision=None,x_calendar = None):
            
            ''' 
            Args:
            -------
            x: 3-th or 4-th order Tensor : [B,N,L] or [B,C,N,L]

                B: batch-size
                C: number of traffic channel (flow, speed, density ...)
                N: number of spatial-units (exemple: 40 subway stations)
                L: length of historical sequence (t-w,t-d,t-6,t-5,t-4,t-3,t-2,t-1)

            1st step: reshape permute input for first st_blocks : [B,C,L,N] 
            
            '''
            # Tackle case where we only want to use the output module (and not the core-model STGCN
            if not (x.numel() == 0):
                # Reshape and permute : [B,N,L] or [B,C,N,L] ->  [B,C,L,N]
                if len(x.size())<4:
                    x = x.unsqueeze(1)
                ### Core model :

                if not x.numel() == 0:
                    #[B,C,N,L] -> [B,C,L,N]
                    x = x.permute(0,1,3,2)
                    # [B,C,L,N] -> [B, C_out, L-4*nb_blocks, N]
                    x = self.st_blocks(x)

                ### ---

            if self.Ko >= 1:
                # Causal_TempConv2D - FC(128,128) -- FC(128,1) -- LN - ReLU --> [B,1,1,N]
                x = self.output(x,x_vision,x_calendar)
            elif self.Ko == 0:
                # [B,C_out,L',N] = [B,1,L',N] actually 
                if self.vision_concatenation_late:
                    # [B,C_out,N,L'] -> [B,C_out,L',N] 
                    x_vision = x_vision.permute(0,1,3,2)
                    # Concat [B,C,L-4*nb_blocks, N] + [B,C_out,L',N]
                    if not (x.numel() == 0):
                        x = torch.concat([x,x_vision],axis=2)
                    else:
                        x = x_vision
                if self.TE_concatenation_late:
                    # [B,C,N,L_calendar]  -> [B,C,L_calendar,N] 
                    x_calendar = x_calendar.permute(0,1,3,2)
                    # Concat [B,C,L-4*nb_blocks, N] + [B,C,L_calendar,N] 
                    if not (x.numel() == 0):
                        x = torch.concat([x,x_calendar],axis=2)
                    else:
                        x = x_calendar

            
                x = self.fc1(x.permute(0, 2, 3, 1))

                x = self.relu(x)
                x = self.fc2(x)
                x = x.permute(0, 3, 1, 2)
            
            #print('x.size: ' ,x.size())
            B = x.size(0)
            x = x.squeeze()
            if B ==1:
                x = x.unsqueeze(0)
            if self.n_vertex == 1:
                x = x.unsqueeze(-1)
            if self.out_dim == 1:
                x = x.unsqueeze(-2)
            #print('x.size: ' ,x.size())

            x = x.permute(0,2,1)

            return x