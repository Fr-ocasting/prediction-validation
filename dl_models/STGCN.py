import torch
import torch.nn as nn

import dl_models.STGCN_layer as layers

# ============================================================
# Inspired by  https://github.com/hazdzz/STGCN/tree/main
# ============================================================


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

    def __init__(self, args, gso, blocks,Ko, n_vertex,args_embedding = None,dic_class2rpz=None,args_vision = None):
        super(STGCN, self).__init__()

        self.out_dim = blocks[-1][-1]
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], args.act_fun, args.graph_conv_type, gso, args.enable_bias, args.dropout,args.enable_padding))
        self.st_blocks = nn.Sequential(*modules)

        self.Ko = Ko


        if self.Ko > 0:
            self.output = layers.OutputBlock(self.Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.act_fun, args.enable_bias, args.dropout)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()


            #self.leaky_relu = nn.LeakyReLU()
            #self.silu = nn.SiLU()
            #self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, x):
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

            # Reshape and permute : [B,N,L] or [B,C,N,L] ->  [B,C,L,N]
            if len(x.size())<4:
                x = x.unsqueeze(1)
            B,C,N,L = x.size()
            x = x.permute(0,1,3,2)
            # ....

            # [B,C,L,N] -> [B, C_out, L-4*nb_blocks, N]
            x = self.st_blocks(x)

            B,C,L,N = x.size() 
            if self.Ko > 1:
                x = self.output(x)
            elif self.Ko == 0:
                # [B,C,L',N] ->  [B,1,L',N]
                x = self.fc1(x.permute(0, 2, 3, 1))
                x = self.relu(x)
                x = self.fc2(x).permute(0, 3, 1, 2)

            x = x.squeeze()
            if B ==1:
                x = x.unsqueeze(0)
            if N == 1:
                x = x.unsqueeze(-1)
            if self.out_dim == 1:
                x = x.unsqueeze(-2)
            x = x.permute(0,2,1)
            return x