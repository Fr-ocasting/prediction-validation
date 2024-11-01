from torch import nn
import torch

class RNN(nn.Module):
    def __init__(self,input_dim,h_dim,C_outs,L, num_layers,bias = True,dropout = 0.0, nonlinearity = 'tanh',batch_first = True,bidirectional = False,lstm = False, gru = False,):
        super().__init__()

        # Parameters
        self.C_outs = C_outs
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.lstm = lstm
        self.h_dim = h_dim

        # Architecture
        if lstm: 
            self.rnn = nn.LSTM(input_size = input_dim, hidden_size = h_dim, num_layers=num_layers,bias=bias,batch_first=batch_first,dropout=dropout,bidirectional=bidirectional)
        elif gru:
            self.rnn = nn.GRU(input_size = input_dim, hidden_size = h_dim, num_layers=num_layers,bias=bias,batch_first=batch_first,dropout=dropout,bidirectional=bidirectional)
        else : 
            self.rnn = nn.RNN(input_size = input_dim, hidden_size = h_dim, num_layers=num_layers,nonlinearity=nonlinearity,bias=bias,batch_first=batch_first,dropout=dropout,bidirectional=bidirectional)  # tanh or ReLU as non linear activation
          
        self.D =  2 if bidirectional else 1
        self.Dense_outs = nn.ModuleList([nn.Linear(c_in,c_out) for c_in,c_out in zip([self.D*h_dim*L]+C_outs[:-1], C_outs)])
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def init_hidden(self,batch_size):
        # Does not care about "batch_first"
        if self.lstm:
            h0 = torch.zeros(self.D*self.num_layers, batch_size, self.h_dim)
            c0 = torch.zeros(self.D*self.num_layers, batch_size, self.h_dim)
            return(h0,c0)
        else:
            h0 = torch.zeros(self.D*self.num_layers, batch_size, self.h_dim)
            return(h0)
    

    def forward(self,x):
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
            x = x.permute(0,2,3,1)
            x = x.reshape(-1,x.size(2),x.size(3))
            permute_back = True

        # Init hidden state


        # Rnn
        if self.lstm:
            h0,c0 = self.init_hidden(x.size(0) if self.batch_first else x.size(1))
            x, (_,_) = self.rnn(x,(h0,c0)) #[B,L,D*H_dim]   
        else:
            h0 = self.init_hidden(x.size(0) if self.batch_first else x.size(1))
            x, _ = self.rnn(x,h0)  #[B,L,D*H_dim]

        # Output Module : 
        x = self.flatten(x)
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