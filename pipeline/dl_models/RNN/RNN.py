from torch import nn
import torch
import ast
from typing import Optional
from torch import Tensor

class RNN(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.input_dim = args.input_dim
        self.h_dim = args.h_dim
        self.C_outs = args.C_outs
        self.L = args.L
        self.num_layers = args.num_layers
        self.out_dim = args.out_dim
        self.bias = args.bias
        self.dropout = float(args.dropout)
        self.lstm = args.lstm if  hasattr(args,'lstm') else False
        self.gru = args.gru if  hasattr(args,'gru') else False
        self.nonlinearity = args.nonlinearity if  hasattr(args,'nonlinearity') else 'tanh'
        self.batch_first = True
        self.device = args.device
        self.out_dim = args.out_dim
        self.vision_concatenation_late = args.vision_concatenation_late if hasattr(args,'vision_concatenation_late') else False
        self.vision_out_dim = args.vision_out_dim if hasattr(args,'vision_out_dim') else None


        # Parameters
        if type(self.C_outs) == str:
            self.C_outs = list(ast.literal_eval(self.C_outs))
        self.C_outs = list(self.C_outs) + [self.out_dim]


        self.bidirectional = bool(args.bidirectional)
        # Architecture
        if self.lstm: 
            self.rnn = nn.LSTM(input_size = self.input_dim, hidden_size = self.h_dim, num_layers=self.num_layers,bias=self.bias,batch_first=self.batch_first,dropout=self.dropout,bidirectional=self.bidirectional)
        elif self.gru:
            self.rnn = nn.GRU(input_size = self.input_dim, hidden_size = self.h_dim, num_layers=self.num_layers,bias=self.bias,batch_first=self.batch_first,dropout=self.dropout,bidirectional=self.bidirectional)
        else : 
            self.rnn = nn.RNN(input_size = self.input_dim, hidden_size = self.h_dim, num_layers=self.num_layers,
            nonlinearity=self.nonlinearity,bias=self.bias,batch_first=self.batch_first,
            dropout=self.dropout,bidirectional=self.bidirectional)  # tanh or ReLU as non linear activation

            # self.rnn.flatten_parameters()
          
        self.D =  2 if self.bidirectional else 1

        ## ======= Tackle Output Module if concatenation with contextual data: 

        L_outs_in = [self.D*self.h_dim*self.L]+self.C_outs[:-1]
        self.vision_concatenation_late = self.vision_concatenation_late
        self.TE_concatenation_late = args.args_embedding.concatenation_late if 'calendar_embedding' in args.dataset_names else False 
        if self.vision_concatenation_late:
            L_outs_in[0] = L_outs_in[0]+ args.vision_out_dim
        if self.TE_concatenation_late:
            self.TE_embedding_dim = args.TE_embedding_dim
            L_outs_in[0] = L_outs_in[0]+ self.TE_embedding_dim
        ## =======
        self.Dense_outs = nn.ModuleList([nn.Linear(c_in,c_out) for c_in,c_out in zip(L_outs_in, self.C_outs)])
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def init_hidden(self,batch_size):
        # Does not care about "batch_first"
        if self.lstm:
            h0 = torch.zeros(self.D*self.num_layers, batch_size, self.h_dim).to(self.device)
            c0 = torch.zeros(self.D*self.num_layers, batch_size, self.h_dim).to(self.device)
            return(h0,c0)
        else:
            h0 = torch.zeros(self.D*self.num_layers, batch_size, self.h_dim).to(self.device)
            return(h0)
    

    def forward(self,x,
                x_vision_early: Optional[Tensor] = None,
                x_vision_late: Optional[Tensor] = None,
                x_calendar: Optional[Tensor] = None,
                contextual: Optional[list[Tensor]]= None,
                ):

        if x_vision_late is not None:
            raise NotImplementedError('x_vision has not been implemented')
        if x_vision_early is not None:
            raise NotImplementedError('x_vision has not been implemented')
        
        # print('x.size: ',x.size())
        # if x_vision is not None:
        #     print('x_vision.size: ',x_vision.size())
        # if x_calendar is not None:
        #     print('x_calendar.size: ',x_calendar.size())

        ''' x.shape : [B,C,N,L]
        
        has to be transformed in [B',L,C] to return [B',L,D*C']  
        Where D = 2 if bidirectional else 1
        '''
        
        if len(x.shape) == 2:
            B,L = x.shape
            C,N = 1,1
        if len(x.shape) == 3:
            B,N,L = x.shape
            C = 1
            permute_back = False
        elif len(x.shape)== 4:
            B,C,N,L = x.shape
            x = x.permute(0,2,3,1)  #[B,N,L,C]
            x = x.reshape(-1,x.size(2),x.size(3))  # [B*N,L,C]
            permute_back = True

        # Init hidden state
        # Rnn  [B*N,L,C] ->  [B*N,L,D*H]
        if self.lstm:
            h0,c0 = self.init_hidden(x.size(0) if self.batch_first else x.size(1))
            x, (_,_) = self.rnn(x,(h0,c0))  
        else:
            h0 = self.init_hidden(x.size(0) if self.batch_first else x.size(1))
            x, _ = self.rnn(x,h0) 

        # Output Module :  [B*N,L,D*H] -> [B*N,L*D*H]
        x = self.flatten(x)

        ## == Concatenation of Contextual Data Before output Module :
        # skip size: [B,H,N,1]
        if self.vision_concatenation_late:
            # [B,1,N,Z] -> [B,N,Z,1]    
            x_vision = x_vision.permute(0,2,3,1)
            #  [B,N,Z,1]-> [B*N,Z]
            x_vision = x_vision.reshape(x_vision.size(0)*x_vision.size(-1),-1)
            # Concat [B*N,L*D*H] + [B*N,Z] ->  [B*N,H*L+Z]
            if not (x.numel() == 0):
                x = torch.cat([x,x_vision],dim=1)
            else:
                x = x_vision
        if self.TE_concatenation_late:

            # print('x_calendar.size before transformation: ',x_calendar.size())
            # [B,1,N,L_calendar]  -> [B,N,L_calendar,1]  
            x_calendar = x_calendar.permute(0,2,3,1)
            # [B,N,L_calendar,1]-> [B*N,L_calendar]
            x_calendar = x_calendar.reshape(x_calendar.size(0)*x_calendar.size(1),-1)          
            # Concat   [B*N,L*D*H] + [B*N,L_calendar]  ->   [B*N,H*L+L_calendar]
            # print('x.size before concat: ',x.size())
            # print('x_calendar.size before concat: ',x_calendar.size())
            if not (x.numel() == 0):
                x = torch.cat([x,x_calendar],dim=1)
            else:
                x = x_calendar
        ## == ...

        for dense_out in self.Dense_outs[:-1]:
            x = self.relu(dense_out(x))
        out = self.Dense_outs[-1](x)    # No activation
        # Reshape 
        if permute_back:
            out = out.reshape(B,N,self.C_outs[-1])
        else:
            raise NotImplementedError('A completer')
        #out = out.squeeze()

        return(out)