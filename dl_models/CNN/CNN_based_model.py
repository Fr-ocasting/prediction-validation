from torch import nn
import torch

# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal import:
from dl_models.TimeEmbedding.time_embedding import TimeEmbedding

class CNN(nn.Module):
    def __init__(self,args,dilation = 1, stride = 1,args_embedding = None):
        super().__init__()
    
        #self.c_out = args.C_outs[-1]
        self.c_out = args.out_dim
        self.dropout = nn.Dropout(args.dropout)

        # List of Conv
        self.Convs = nn.ModuleList([nn.Conv1d(c_in_, c_out_, args.kernel_size,padding=args.padding,dilation=dilation) for c_in_,c_out_ in zip([args.c_in]+args.H_dims[:-1], args.H_dims)])

        # Calculate the last dim of the sequence : 
        l_out_add = (2*args.padding - dilation*(args.kernel_size[0]-1) -1)/stride + 1

        if (args_embedding is not None) and len(vars(args_embedding))>0:
                if args_embedding.position == 'input':
                    L = args.L+args_embedding.embedding_dim
                    l_out = int(L/stride**len(args.H_dims) + sum([l_out_add/stride**k for k in range(len(args.H_dims))]))

                if args_embedding.position == 'output':
                    l_out = int( args.L/stride**len(args.H_dims) + sum([l_out_add/stride**k for k in range(len(args.H_dims))]))
                    l_out = l_out+args_embedding.embedding_dim

        else:
            L = args.L
            l_out = int(L/stride**len(args.H_dims) + sum([l_out_add/stride**k for k in range(len(args.H_dims))])) 
            
        self.l_out = l_out

        # ... 

        # Activation, Flatten and Regularization : 
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Output Module (traditionnaly 1 or 2 linear layers)
        self.Dense_outs = nn.ModuleList([nn.Linear(c_in,c_out) for c_in,c_out in zip([l_out*args.H_dims[-1]]+args.C_outs, args.C_outs+[self.c_out])])
            
    def forward(self,x):
        '''
        Pass input through a 2 layer CNN
             --------  --------    -------    -------    ------    ------
        X -- TempConv--TempConv -- Flatten -- Reshape -- Dense1 -- Dense2 --> Y [B,N,Cout]   (with Cout = 1 usually)
             --------  --------    -------    -------    ------    ------

        Input: 
        ------
        x : shape [B,1,N,L] or [B,C,N,L]

        Output:
        ------
        x.shape [B,N,1]

        '''
        if x.dim() == 3:
            B,N,L = x.shape
            C = 1
            x = x.unsqueeze(2)
            x = x.reshape(B*N,C,L)
        elif x.dim() == 4:
            B,C,N,L = x.shape
            x = x.reshape(B*N,C,L)
        else:
            raise NotImplemented(f'Dimension {x.dim()} has not been implemented')

        # Conv Layers :        
        for conv in self.Convs:
            x = self.dropout(self.relu(conv(x)))

        # Flatten :
        x = self.flatten(x)
        # Output Module : 
        for dense_out in self.Dense_outs[:-1]:
            x = self.dropout(self.relu(dense_out(x)))

        x = self.Dense_outs[-1](x)    # No activation
        # Reshape 
        x = x.reshape(B,N,self.c_out)
        return(x)