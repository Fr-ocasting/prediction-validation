from torch import nn
import torch

class rnn_perso(nn.Module):
    def __init__(self,c_in,h_dim,c_out, num_layers, nonlinearity = 'tanh',bias = True,batch_first = True,dropout = 0.0,bidirectional = False,lstm = False, gru = False):
        super().__init__()

        # Parameters
        self.c_in = c_in
        self.h_dim = h_dim
        self.c_out = c_out
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.lstm = lstm
        self.gru = gru
        # Architecture
        if lstm: 
            self.rnn = nn.LSTM(input_size = c_in, hidden_size = h_dim, num_layers=num_layers,bias=bias,batch_first=batch_first,dropout=dropout,bidirectional=bidirectional)
        elif gru:
            self.rnn = nn.GRU(input_size = c_in, hidden_size = h_dim, num_layers=num_layers,bias=bias,batch_first=batch_first,dropout=dropout,bidirectional=bidirectional)
        else : 
            self.rnn = nn.RNN(input_size = c_in, hidden_size = h_dim, num_layers=num_layers,nonlinearity=nonlinearity,bias=bias,batch_first=batch_first,dropout=dropout,bidirectional=bidirectional)  # tanh or ReLU as non linear activation
          
        self.D =  2 if self.bidirectional else 1
        self.dense_out1 = nn.Linear(self.D*h_dim,32)
        self.dense_out2 = nn.Linear(32,c_out)
        self.relu = nn.ReLU()

    def init_hidden(self,batch_size):
        # Does not care about "batch_first"
        if self.lstm:
            h0 = torch.zeros(self.D*self.num_layers, batch_size, self.h_dim)
            c0 = torch.zeros(self.D*self.num_layers, batch_size, self.h_dim)
            return((h0,c0))
        else:
            h0 = torch.zeros(self.D*self.num_layers, batch_size, self.h_dim)
            return(h0)
    

    def forward(self,x):
        # Init hidden state
        h0 = self.init_hidden(x.size(0) if self.batch_first else x.size(1))

        # Rnn
        x, hn = self.rnn(x,h0)  #[B,L,D*H_dim]

        # Output 
        x = self.relu(self.dense_out1(x)) # Linear only on the last dimension (here = h_dim)
        x = self.relu(self.dense_out2(x))

        return(x)