from torch import nn
import torch

from dl_models.time_embedding import TimeEmbedding

class CNN(nn.Module):
    def __init__(self,c_in, H_dims, C_outs, kernel_size, L, padding = 0, dilation = 1, stride = 1, dropout = 0.0,time_embedding_args = None):
        super().__init__()
    
        self.c_in = c_in
        self.C_outs = C_outs
        self.H_dims = H_dims
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.name = f"CNN_{len(H_dims)}layer_h{'_'.join(list(map(str,H_dims)))}"
        self.dropout = nn.Dropout(dropout)

        # List of Conv
        self.Convs = nn.ModuleList([nn.Conv1d(c_in_, c_out_, kernel_size,padding=padding,dilation=dilation) for c_in_,c_out_ in zip([c_in]+H_dims[:-1], H_dims)])

        # Calculate the last dim of the sequence : 
        if time_embedding_args is not None:
                L = L+time_embedding_args.embedding_dim

        l_out_add = (2*padding - dilation*(kernel_size[0]-1) -1)/stride + 1
        l_out = int(L/stride**len(H_dims) + sum([l_out_add/stride**k for k in range(len(H_dims))]))
        self.l_out = l_out

        # Activation, Flatten and Regularization : 
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Output Module (traditionnaly 1 or 2 linear layers)
        self.Dense_outs = nn.ModuleList([nn.Linear(c_in,c_out) for c_in,c_out in zip([l_out*H_dims[-1]]+C_outs[:-1], C_outs)])

        if time_embedding_args is not None:
            self.Tembedding = TimeEmbedding(time_embedding_args.Encoded_dims,time_embedding_args.embedding_dim)
            self.Tembedding_position = time_embedding_args.position
    
    def forward(self,x,time_elt = None):
        if len(x.shape) == 3:
            B,N,L = x.shape
            x = x.unsqueeze(2)
            x = x.reshape(B*N,1,L)
        if len(x.shape) == 4:
            print('! Be sure x.shape = [B,C,N,L]')
            B,C,N,L = x.shape
            x.reshape(B*N,C,L)

        if time_elt is not None:
            if self.Tembedding_position == 'input':
                time_elt = self.Tembedding(time_elt)
                time_elt = time_elt.repeat(B*N,C,1)

                x = torch.cat([x,time_elt],dim = -1)

        # Conv Layers :        
        for conv in self.Convs:
            x = self.dropout(self.relu(conv(x)))

        if time_elt is not None:
            if self.Tembedding_position == 'input':
                time_elt = self.Tembedding(time_elt)
                time_elt = time_elt.repeat(B*N,C,1)

                x = torch.cat([x,time_elt],dim = -1)

        # Flatten :
            x = self.flatten(x)
        # Output Module : 
        for dense_out in self.Dense_outs[:-1]:
            x = self.dropout(self.relu(dense_out(x)))

        x = self.Dense_outs[-1](x)    # No activation
        # Reshape 
        x = x.reshape(B,N,self.C_outs[-1])
        return(x)
    



class cnn_perso(nn.Module):
    def __init__(self,c_in, h_dim, c_out, kernel_size, L, padding = 0, time_embedding_args = None):
        super().__init__()

        dim_last_conv = 2*h_dim

        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.name = f'CNN_h{h_dim}_L{L}_2layer'
        h_out = (dim_last_conv*(L-2))

        self.conv1 = nn.Conv1d(c_in, h_dim, kernel_size,padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(h_dim, 2*h_dim, kernel_size, padding=padding)
        self.flatten = nn.Flatten()
        
        self.dense_out1 = nn.Linear(h_out, 32)
        self.dense_out2 = nn.Linear(32, c_out)

        if time_embedding_args is not None:
            self.Tembedding = TimeEmbedding(time_embedding_args.Encoded_dims,time_embedding_args.embedding_dim)
            self.Tembedding_position = time_embedding_args.position

    def forward(self,x,time_elt = None):
        if len(x.shape) == 3:
            B,N,L = x.shape
            x = x.unsqueeze(2)
            x = x.reshape(B*N,1,L)
        
        if x_time is not None:
            if self.Tembedding_position == 'input':
                x_time = self.Tembedding(time_elt)
                x_time.repeat(B*N,1,1)

                x = torch.cat([x,time_elt],dim = -1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        if x_time is not None:
            if self.Tembedding_position == 'output':
                x_time = self.Tembedding(time_elt)
                x_time.repeat(B*N,1,1)
                x = torch.cat([x,time_elt],dim = -1)

        x = self.flatten(x)
        x = self.dense_out1(x)
        x = self.dense_out2(x)
        x = x.reshape(B,N,self.c_out)


        return(x)