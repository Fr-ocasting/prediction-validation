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
from dl_models.time_embedding import TimeEmbedding

class CNN(nn.Module):
    def __init__(self,args,kernel_size,dilation = 1, stride = 1,args_embedding = None,dic_class2rpz=None):
        super().__init__()
    
        self.c_out = args.C_outs[-1]
        self.dropout = nn.Dropout(args.dropout)

        # List of Conv
        self.Convs = nn.ModuleList([nn.Conv1d(c_in_, c_out_, kernel_size,padding=args.padding,dilation=dilation) for c_in_,c_out_ in zip([args.c_in]+args.H_dims[:-1], args.H_dims)])

        # Calculate the last dim of the sequence : 
        l_out_add = (2*args.padding - dilation*(kernel_size[0]-1) -1)/stride + 1

        if args_embedding is not None:
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
        self.Dense_outs = nn.ModuleList([nn.Linear(c_in,c_out) for c_in,c_out in zip([l_out*args.H_dims[-1]]+args.C_outs[:-1], args.C_outs)])

        if args_embedding is not None:
            mapping_tensor = torch.tensor([(week[0], time[0][0], time[0][1]) for _, (week, time) in sorted(dic_class2rpz[args.calendar_class].items())]).to(args.device)
            self.Tembedding = TimeEmbedding(args_embedding.nb_words_embedding,args_embedding.embedding_dim,args.type_calendar,mapping_tensor)
            self.Tembedding_position = args_embedding.position

            
    def forward(self,x,time_elt = None):
        if len(x.shape) == 3:
            B,N,L = x.shape
            C = 1
            x = x.unsqueeze(2)
            x = x.reshape(B*N,C,L)
        if len(x.shape) == 4:
            B,C,N,L = x.shape
            x.reshape(B*N,C,L)


        if time_elt is not None:  #(week,hour,day)
            if self.Tembedding_position == 'input':

                time_elt = self.Tembedding(time_elt)   # [B,1] -> [B,embedding_dim]
                time_elt = time_elt.repeat(N*C,1).reshape(B*N,C,-1)   # [B,embedding_dim] -> [B*N,C,embedding_dim]
                x = torch.cat([x,time_elt],dim = -1)

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