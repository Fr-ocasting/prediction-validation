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
from pipeline.dl_models.TimeEmbedding.time_embedding import TimeEmbedding

class CNN(nn.Module):
    def __init__(self,args,dilation = 1, stride = 1,L_add = 0,
                 vision_concatenation_late = False,TE_concatenation_late = False,vision_out_dim = None,TE_embedding_dim = None):
        super().__init__()
    
        #self.c_out = args.C_outs[-1]
        self.c_out = args.out_dim
        self.dropout = nn.Dropout(args.dropout)

        # List of Conv
        self.Convs = nn.ModuleList([nn.Conv1d(c_in_, c_out_, args.kernel_size,padding=args.padding,dilation=dilation) for c_in_,c_out_ in zip([args.c_in]+args.H_dims[:-1], args.H_dims)])

        # Calculate the last dim of the sequence : 
        l_out_add = (2*args.padding - dilation*(args.kernel_size[0]-1) -1)/stride + 1
        if False : 
            if args_embedding.position == 'output':
                l_out = int( args.L/stride**len(args.H_dims) + sum([l_out_add/stride**k for k in range(len(args.H_dims))]))
                l_out = l_out+args_embedding.embedding_dim
                raise NotImplementedError
        if L_add != 0:
            L = args.L + L_add
        else : 
            L = args.L
        l_out = int(L/stride**len(args.H_dims) + sum([l_out_add/stride**k for k in range(len(args.H_dims))]))
            
        self.l_out = l_out

        # ... 

        # Activation, Flatten and Regularization : 
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()


        ## ======= Tackle Output Module if concatenation with contextual data: 
        L_outs_in = [l_out*args.H_dims[-1]]+args.C_outs
        self.vision_concatenation_late = vision_concatenation_late
        self.TE_concatenation_late = args.args_embedding.concatenation_late if 'calendar_embedding' in args.dataset_names else False 
        if self.vision_concatenation_late:
            L_outs_in[0] = L_outs_in[0]+ vision_out_dim
        if self.TE_concatenation_late:
            L_outs_in[0] = L_outs_in[0]+ TE_embedding_dim
        ## =======
        # Output Module (traditionnaly 1 or 2 linear layers)
        self.Dense_outs = nn.ModuleList([nn.Linear(c_in,c_out) for c_in,c_out in zip(L_outs_in, args.C_outs+[self.c_out])])
            
    def forward(self,x,x_vision=None,x_calendar = None):
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
        # x [B,N,L] or [B,N,C,L] -> [B*N,C,L]
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

        # Conv Layers :    [B*N,C,L] -> [B*N,H,L]        
        for conv in self.Convs:
            x = self.dropout(self.relu(conv(x)))


        # Flatten :   [B*N,H,L] -> [B*N,H*L]
        x = self.flatten(x)

        ## == Concatenation of Contextual Data Before output Module :
        # skip size: [B,H,N,1]
        if self.vision_concatenation_late:
            # [B,1,N,Z] -> [B,N,Z,1]    
            x_vision = x_vision.permute(0,2,3,1)
            #  [B,N,Z,1]-> [B*N,Z]
            x_vision = x_vision.reshape(x_vision.size(0)*x_vision.size(-1),-1)
            # Concat [B*N,H*L] + [B*N,Z] ->  [B*N,H*L+Z]
            if not (x.numel() == 0):
                x = torch.cat([x,x_vision],dim=1)
            else:
                x = x_vision
        if self.TE_concatenation_late:
            # [B,1,N,L_calendar]  -> [B,N,L_calendar,1]  
            x_calendar = x_calendar.permute(0,2,3,1)
            # [B,N,L_calendar,1]-> [B*N,L_calendar]
            x_calendar = x_calendar.reshape(x_calendar.size(0)*x_calendar.size(-1),-1)          
            # Concat   [B*N,H*L] + [B*N,L_calendar]  ->   [B*N,H*L+L_calendar]
            if not (x.numel() == 0):
                x = torch.cat([x,x_calendar],dim=1)
            else:
                x = x_calendar
        ## == ...


        # Output Module : 
        for dense_out in self.Dense_outs[:-1]:
            x = self.dropout(self.relu(dense_out(x)))

        x = self.Dense_outs[-1](x)    # No activation
        # Reshape 
        x = x.reshape(B,N,self.c_out)
        return(x)