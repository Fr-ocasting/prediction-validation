from torch import nn
import torch

class RNN(nn.Module):
    def __init__(self,input_dim,h_dim,C_outs,L, num_layers,out_dim, bias = True,
                 dropout = 0.0, nonlinearity = 'tanh',batch_first = True,
                 bidirectional = False,lstm = False, gru = False, device = None,
                 vision_concatenation_late = False,TE_concatenation_late = False,vision_out_dim = None,TE_embedding_dim = None):
        super().__init__()

        # Parameters
        self.C_outs = C_outs + [out_dim]
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.lstm = lstm
        self.h_dim = h_dim
        self.device = device

        # Architecture
        if lstm: 
            self.rnn = nn.LSTM(input_size = input_dim, hidden_size = h_dim, num_layers=num_layers,bias=bias,batch_first=batch_first,dropout=dropout,bidirectional=bidirectional)
        elif gru:
            self.rnn = nn.GRU(input_size = input_dim, hidden_size = h_dim, num_layers=num_layers,bias=bias,batch_first=batch_first,dropout=dropout,bidirectional=bidirectional)
        else : 
            self.rnn = nn.RNN(input_size = input_dim, hidden_size = h_dim, num_layers=num_layers,nonlinearity=nonlinearity,bias=bias,batch_first=batch_first,dropout=dropout,bidirectional=bidirectional)  # tanh or ReLU as non linear activation
          
        self.D =  2 if bidirectional else 1

        ## ======= Tackle Output Module if concatenation with contextual data: 
        L_outs_in = [self.D*h_dim*L]+self.C_outs[:-1]
        self.vision_concatenation_late = vision_concatenation_late
        self.TE_concatenation_late = TE_concatenation_late
        if self.vision_concatenation_late:
            L_outs_in[0] = L_outs_in[0]+ vision_out_dim
        if self.TE_concatenation_late:
            L_outs_in[0] = L_outs_in[0]+ TE_embedding_dim
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
    

    def forward(self,x,x_vision=None,x_calendar = None):

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