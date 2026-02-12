#----------------------------------------------------#
#
#   File       : VariableSelectionNetwork.py
#   Author     : Soham Deshpande
#   Date       : January 2022
#   Description: VariableSelection Network
#
#
#
# ----------------------------------------------------#

import torch
import torch.nn as nn

class VariableSelectionNetwork(nn.Module):

    """
    VariableSelectionNetwork

    VRN(x) = GRN(x) x GRN(x) x Softmax(GRN(x))

    Args:
        int input_size : Size of input tensor
        int hidden_size: Size of the hidden layer
        int output_size: Size of the output layer
        float dropout  : Fraction between 0 and 1 showing the dropout rate

    """

    def __init__(self, input_size, output_size, hidden_size,dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout    = dropout
        self.flattened_inputs = GRN(self.output_size*self.input_size,
                                                     self.hidden_size, self.output_size,
                                                     self.dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.transformed_inputs = nn.ModuleList([GRN(
                self.input_size, self.hidden_size, self.hidden_size,
                self.dropout) for i in range(self.output_size)])

    def forward(self, embedding, context=None):
        """
        Args:
          embedding (torch.tensor): Entity embeddings for categorical variables and linear
                     transformations for continuous variables.
          context (torch.tensor): The context is obtained from a static covariate encoder and
                   is naturally omitted for static variables as they already
                   have access to this
        """

        # Generation of variable selection weights
        sparse_weights = self.flattened_inputs(embedding, context)
        if self.is_temporal:
            sparse_weights = self.softmax(sparse_weights).unsqueeze(2)
        else:
            sparse_weights = self.softmax(sparse_weights).unsqueeze(1)

        # Additional non-linear processing for each feature vector
        transformed_embeddings = torch.stack(
            [self.transformed_inputs[i](embedding[
                Ellipsis, i*self.input_size:(i+1)*self.input_size]) for i in range(self.output_size)], axis=-1)

        # Processed features are weighted by their corresponding weights and combined
        combined = transformed_embeddings*sparse_weights
        combined = combined.sum(axis=-1)

        return combined, sparse_weight
    
class GLU(nn.Module):
    """
    Gated Linear Unit

    GLU(x,y) = multiply(x, sigmoid(y))

    Args:
        int input size: Defines the size of the input matrix and output size of
        the gate

    """

    def __init__(self,input_size):
        super().__init__()
        #input
        self.x = nn.Linear(input_size,input_size) # construct matrix 1

        #Gate
        self.y = nn.Linear(input_size, input_size) # construct matrix 2
        self.sigmoid = nn.Sigmoid() # construct sigmoid function


    def forward(self,a):
        """
        Args:
            float(tensor) a: Tensor that passes through the gate

        """
        gate = self.sigmoid(self.y(x))
        x = self.x(a)

        return torch.mul(gate, x) #multiply both tensors together
    
class TemporalLayer(nn.Module):
    def __init__(self, module):
        super().__init__()
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        An implitation of the TimeDistributed layer used in Tensorflow.
        Applied at every temporal slice of an input

        """
        self.module = module


    def forward(self, x):
        """
        Args:
            x (torch.tensor): Tensor with time steps to pass through the same layer.
        """
        t, n = x.size(0), x.size(1)
        x = x.reshape(t * n, -1)
        x = self.module(x)
        x = x.reshape(t, n, x.size(-1))

        return x
    
class GRN(nn.Module):

    """
    Gated Residual Network

    GRN(x) = LayerNorm(a + GLU(Linear(a)))

    Args:
       int   input_size  : Size of the input tensor
       int   hidden_size : Size of the hidden layer
       int   output_size : Size of the output layer
       float   dropout   : Fraction between 0 and 1 showing the dropout rate
       int   context_size: Size of the context vector
       bool  is_temporal : Decides if the Temporal Layer has to be used or not


    This unit controls how much of the original input is used. It can skip over
    layers where the GLU output might be close to 0.
    When there is no context vector present, the GRN will treat the input as 0.
    """

    def __init__(self, input_size,hidden_size, output_size, dropout,
            context_size=None, is_temporal=True):
        super().__init__()
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.output_size  = output_size
        self.dropout      = dropout
        self.is_temporal  = is_temporal
        self.context_size = context_size

        if self.input_size != self.output_size:
                self.skip_layer = TemporalLayer(nn.Linear(self.input_size,
                    self.output_size))

            # Context vector c
        if self.context_size != None:
            self.c = TemporalLayer(nn.Linear(self.context_size,
                self.hidden_size, bias=False))

        # Dense & ELU
        self.dense1 = TemporalLayer(nn.Linear(self.input_size,
            self.hidden_size))
        self.elu = nn.ELU()

        # Dense & Dropout
        self.dense2 = TemporalLayer(nn.Linear(self.hidden_size,
            self.output_size))
        self.dropout = nn.Dropout(self.dropout)

        # Gate, Add & Norm
        self.gate = TemporalLayer(GLU(self.output_size))
        self.layer_norm = TemporalLayer(nn.BatchNorm1d(self.output_size))

    def forward(self, x):
        a = nn.ELU(self.c(x))
        a = self.dropout(self.dense2(a))

        a = self.gate(a)

        if(self.skip != None):
            return self.norm(self.skip(x) + a)
        return self.norm(x + a)


#vsn = VariableSelectionNetwork(2,2,4,0.1)
#x = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
#y = torch.tensor(np.array([[1, 2, 3]]))

#print(vsn.forward(1))