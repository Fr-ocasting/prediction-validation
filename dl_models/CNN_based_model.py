from torch import nn
import torch

from dl_models.time_embedding import TimeEmbedding

class CNN(nn.Module):
    def __init__(self,c_in, H_dims, C_outs, kernel_size, L, padding = 0, dilation = 1, stride = 1, dropout = 0.0,args_embedding = None):
        super().__init__()
    
        self.c_out = C_outs[-1]
        self.dropout = nn.Dropout(dropout)

        # List of Conv
        self.Convs = nn.ModuleList([nn.Conv1d(c_in_, c_out_, kernel_size,padding=padding,dilation=dilation) for c_in_,c_out_ in zip([c_in]+H_dims[:-1], H_dims)])

        # Calculate the last dim of the sequence : 
        if args_embedding is not None:
                if args_embedding.position == 'input':
                    L = L+args_embedding.embedding_dim

        l_out_add = (2*padding - dilation*(kernel_size[0]-1) -1)/stride + 1
        l_out = int(L/stride**len(H_dims) + sum([l_out_add/stride**k for k in range(len(H_dims))]))

        if args_embedding is not None:
                if args_embedding.position == 'output':
                    l_out = l_out+args_embedding.embedding_dim

        self.l_out = l_out
        # ... 

        # Activation, Flatten and Regularization : 
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Output Module (traditionnaly 1 or 2 linear layers)
        self.Dense_outs = nn.ModuleList([nn.Linear(c_in,c_out) for c_in,c_out in zip([l_out*H_dims[-1]]+C_outs[:-1], C_outs)])

        if args_embedding is not None:
            self.Tembedding = TimeEmbedding(args_embedding.nb_words_embedding,args_embedding.embedding_dim)
            self.Tembedding_position = args_embedding.position
    
    def forward(self,x,time_elt = None):
        if len(x.shape) == 3:
            B,N,L = x.shape
            C = 1
            x = x.unsqueeze(2)
            x = x.reshape(B*N,C,L)
        if len(x.shape) == 4:
            print('! Be sure input shape = [B,C,N,L]')
            B,C,N,L = x.shape
            x.reshape(B*N,C,L)


        if time_elt is not None:  #(week,hour,day)
            if self.Tembedding_position == 'input':
                time_elt = self.Tembedding(time_elt)   # [B,1] -> [B,embedding_dim]
                time_elt = time_elt.repeat(N*C,1).reshape(B*N,C,-1)   # [B,embedding_dim] -> [B*N,C,embedding_dim]
                x = torch.cat([x,time_elt],dim = -1)
        else :
            print("model ne prend pas en compte l'embedding de temps" )

        # Conv Layers :        
        for conv in self.Convs:
            x = self.dropout(self.relu(conv(x)))

        if time_elt is not None:
            if self.Tembedding_position == 'output':
                BN,C,L = x.size()
                time_elt = self.Tembedding(time_elt)
                time_elt = time_elt.repeat(N*C,1).reshape(BN,C,-1)

                x = torch.cat([x,time_elt],dim = -1)

        # Flatten :
        x = self.flatten(x)
        # Output Module : 
        for dense_out in self.Dense_outs[:-1]:
            x = self.dropout(self.relu(dense_out(x)))

        x = self.Dense_outs[-1](x)    # No activation
        # Reshape 
        x = x.reshape(B,N,self.c_out)
        return(x)