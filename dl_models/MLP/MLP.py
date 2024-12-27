from torch import nn
import torch

class MLP_output(nn.Module):
    '''
    Weird Class allowing to use 'MLP_embedding' as an output module in 'full model'.

    2 Layer Perceptron to capture calendar dependencies.
    inputs:
    --------
    x : enhanced data. have to pass by a MLP for prediction.
    extracted_feature : useless here, supposed to be already used 
    time_elt : useless here, supposed to be already used 

    '''
    def __init__(self,input_dim,out_h_dim,n_vertex,embedding_dim,multi_embedding,dropout):
        super(MLP_output, self).__init__()
        self.mlp = MLP_embedding(input_dim,out_h_dim,n_vertex,embedding_dim,multi_embedding,dropout)

    def forward(self,x,extracted_feature,time_elt): 
        x  = self.mlp(None,x)
        return(x)


class MLP_embedding(nn.Module):
    '''
    2 Layer Perceptron to capture calendar dependencies.
    inputs:
    --------
    Concatenation of embedded vector. 
    '''
    def __init__(self,input_dim,out_h_dim,n_vertex,embedding_dim,multi_embedding,dropout):
        super(MLP_embedding, self).__init__()
        self.output1 = nn.Linear(input_dim,out_h_dim)

        if multi_embedding:
            self.output2 = nn.Linear(out_h_dim,n_vertex*embedding_dim)
        else:
            self.output2 = nn.Linear(out_h_dim,embedding_dim)        

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,z): 
        '''
        args:
        ------
        x is None (only consider calendar embedding here, does not mix it with trafic flow)
        z : concatenation of all the calendar embedding. 2-th order Tensor [B,z]
        '''
        z = self.dropout(self.output1(z))
        z = self.output2(self.relu(z))
        return(z)