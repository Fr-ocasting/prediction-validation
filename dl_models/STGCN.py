import torch
import torch.nn as nn

import dl_models.STGCN_layer as layers

from dl_models.time_embedding import TimeEmbedding

class STGCNChebGraphConv(nn.Module):
    # STGCNChebGraphConv contains 'TGTND TGTND TNFF' structure
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

    def __init__(self, args, blocks, n_vertex,args_embedding = None,dic_class2rpz=None):
        super(STGCNChebGraphConv, self).__init__()
        modules = []
        self.args = args
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], args.act_fun, args.graph_conv_type, args.gso, args.enable_bias, args.dropout,args.enable_padding))
        self.st_blocks = nn.Sequential(*modules)

        Ko = args.L - (len(blocks) - 3) * 2 * (args.Kt - 1)
        if args.enable_padding:
            Ko = args.L if args.L > 0 else 1

        if args_embedding is not None:
            Ko = Ko + args_embedding.embedding_dim

        self.Ko = Ko
    
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.act_fun, args.enable_bias, args.dropout)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU()
            self.silu = nn.SiLU()
            self.dropout = nn.Dropout(p=args.dropout)


        if args_embedding is not None:
            mapping_tensor = torch.tensor([(week[0], time[0][0], time[0][1]) for _, (week, time) in sorted(dic_class2rpz.items())]).to(args.device)
            self.Tembedding = TimeEmbedding(args_embedding.nb_words_embedding,args_embedding.embedding_dim,args.type_calendar,mapping_tensor)
            self.Tembedding_position = args_embedding.position

    def forward(self, x, time_elt = None):
        if len(x.size())<4:
            x = x.unsqueeze(1)

        B,C,N,L = x.size()

        if time_elt is not None:
            if self.Tembedding_position == 'input':
                time_elt = self.Tembedding(time_elt)   # [B,1] -> [B,embedding_dim]
                time_elt = time_elt.repeat(N*C,1).reshape(B,C,N,-1)   # [B,embedding_dim] -> [B,C,embedding_dim,N]
                x = torch.cat([x,time_elt],dim = -1)

        #x : [B,C,N,L]
        # st_blocks inputs: [B,C,L,N]. Therefore, we need to permute: 
        x = x.permute(0,1,3,2)
        x = self.st_blocks(x)

        B,C,L,N = x.size()
        if time_elt is not None:
            if self.Tembedding_position == 'output': 
                time_elt = self.Tembedding(time_elt)   # [B,1] -> [B,embedding_dim]
                time_elt = time_elt.repeat(N*C,1).reshape(B,C,-1,N)   # [B,embedding_dim] -> [B,C,embedding_dim,N]
                x = torch.cat([x,time_elt],dim = 2)

        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        x = x.squeeze()
        if B ==1:
            x = x.unsqueeze(0)
        if N == 1:
            x = x.unsqueeze(-1)
        x = x.permute(0,2,1)
        return x

class STGCNGraphConv(nn.Module):
    # STGCNGraphConv contains 'TGTND TGTND TNFF' structure
    # GraphConv is the graph convolution from GCN.
    # GraphConv is not the first-order ChebConv, because the renormalization trick is adopted.
    # Be careful about over-smoothing.
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, args, blocks, n_vertex,args_embedding = None,dic_class2rpz =None):
        super(STGCNGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], args.act_fun, args.graph_conv_type, args.gso, args.enable_bias, args.dropout,args.enable_padding))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.L - (len(blocks) - 3) * 2 * (args.Kt - 1)

        # Ajout perso, dans le cas ou Ko < 0, on a 'enable padding' obligatoire 
        # ----
        if args.enable_padding:
            Ko = args.L if args.L > 0 else 1
        if args_embedding is not None:
            Ko = Ko + args_embedding.embedding_dim
        # ----
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.act_fun, args.enable_bias, args.dropout)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU()
            self.silu = nn.SiLU()
            self.do = nn.Dropout(p=args.dropout)

        if args_embedding is not None:
            #mapping_tensor = torch.tensor([(week[0], time[0][0], time[0][1], bank_holiday) for _, (week, time, bank_holiday) in sorted(dic_class2rpz.items())]).to(args.device)
            mapping_tensor = torch.tensor([(week[0], time[0][0], time[0][1]) for _, (week, time) in sorted(dic_class2rpz.items())]).to(args.device)
            self.Tembedding = TimeEmbedding(args_embedding.nb_words_embedding,args_embedding.embedding_dim,args.type_calendar,mapping_tensor)
            self.Tembedding_position = args_embedding.position

    def forward(self, x,time_elt = None):
        if len(x.size())<4:
            x = x.unsqueeze(1)
        B,C,N,L = x.size()
    
        if time_elt is not None:
            if self.Tembedding_position == 'input':
                time_elt = self.Tembedding(time_elt)   # [B,1] -> [B,embedding_dim]
                time_elt = time_elt.repeat(N*C,1).reshape(B,C,N,-1)   # [B,embedding_dim] -> [B,C,embedding_dim,N]
                x = torch.cat([x,time_elt],dim = -1)

        #x : [B,C,N,L]
        # st_blocks inputs: [B,C,L,N]. Therefore, we need to permute: 
        x = x.permute(0,1,3,2)
        x = self.st_blocks(x)
        # st_blocks outputs: [B, C_out, L-4*nb_blocks, N])
        B,C,L,N = x.size() 
        if time_elt is not None:
            if self.Tembedding_position == 'output':
                time_elt = self.Tembedding(time_elt)   # [B,1] -> [B,embedding_dim]
                time_elt = time_elt.repeat(N*C,1).reshape(B,C,-1,N)   # [B,embedding_dim] -> [B,C,embedding_dim,N]
                x = torch.cat([x,time_elt],dim = 2)
        
        
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        x = x.squeeze()
        if B ==1:
            x = x.unsqueeze(0)
        if N == 1:
            x = x.unsqueeze(-1)
        x = x.permute(0,2,1)
        return x