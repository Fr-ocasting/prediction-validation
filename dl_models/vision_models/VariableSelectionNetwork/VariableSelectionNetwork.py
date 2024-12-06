import torch
import torch.nn as nn
import math
# ============================ ======================== ============================
# ============================ VariableSelectionNetwork ============================
class FastGLU(nn.Module):
    def __init__(self,input_size):
        super(FastGLU,self).__init__()
        self.input_size = input_size
        self.dense = nn.Linear(input_size,input_size*2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        ''' 
        x : n-th order tensensor 
        '''
        x = self.dense(x)
        out = x[...,:self.input_size] * self.sigmoid(x[...,self.input_size:])
        return(out)


class GRN(nn.Module):
    def __init__(self,input_size,grn_h_dim,grn_out_dim,contextual_static_dim,dropout):
        super(GRN,self).__init__()
        self.input_size = input_size
        self.grn_out_dim = grn_out_dim
        if self.input_size != self.grn_out_dim:
            self.align = nn.Linear(self.input_size,self.grn_out_dim)
        self.dense1 = nn.Linear(self.input_size,grn_h_dim) 
        if contextual_static_dim is not None:
            self.dense_contextual = nn.Linear(contextual_static_dim,grn_h_dim)
        self.dense2 = nn.Linear(grn_h_dim,self.grn_out_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        # Gate:
        self.gate = FastGLU(grn_out_dim)
        # LayerNorm
        self.layer_norm = nn.LayerNorm(grn_out_dim)
        #self.layer_norm = nn.BatchNorm1d(grn_out_dim)
    def forward(self,x,x_c=None):
        '''
        Inputs: 
        -------
        x : 3-th order tensor:  [B,C,input_size] 
        >>> B: Batch-Size
        >>> C: Channel dimension -> Number of POIs * Number of Application 
        >>> input_size : usually sequence length L

        x_c : 2-th order tensor:  [B,z_contextual] 
        >>> z_contextual: dimension of contextual embedding for static information

        Ouputs: 
        -------
        output : 3-th order tensor:  [B,C,grn_out_dim] 
        '''
        x_init = x
        if self.input_size != self.grn_out_dim:
            x_init = self.align(x_init)

        ## Forward backbone model 

        #[B,C,h_dim] -> [B,C,h_dim]
        x = self.dense1(x)
        if x_c is not None:
            #[B,z_contextual] -> [B,h_dim] 
            contextual_info = self.dense_contextual(x_c)

            # PAS BON ON VEUT IDEALEMENT QUE CE SOIT DIFFERENT  (same contextual information added to every POIs)
            # [B,C,h_dim] + [B,h_dim] -> [B,C,h_dim]   
            x  = x+contextual_info

        x = self.elu(x)
        x = self.dense2(x)
        x = self.dropout(x)

        # Gate
        x = self.gate(x)
        
        
        # Add & LayerNorm
        x = x+x_init  # Add
        x = self.layer_norm(x)  # LayerNorm
        return(x)

class Attn_weight_POI(nn.Module):
    def __init__(self,input_size,grn_h_dim,nb_channels,contextual_static_dim,dropout):
        super(Attn_weight_POI,self).__init__()
        self.grn = GRN(input_size,grn_h_dim,nb_channels,contextual_static_dim,dropout)  # (input_size*nb_channels, grn_h_dim, nb_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x,x_c):
        '''
        Inputs: 
        -------
        x : 3-th order tensor:  [B,C,L]
        x_c : 2-th order tensor:  [B,z_contextual]

        Ouputs: 
        -------
        output : 2-th order tensor:  [B,C] 
        '''
        out_grn = self.grn(x,x_c)
        attn_weight = self.softmax(out_grn)
        return(attn_weight)



class VariableSelectionNetwork(nn.Module):
    def __init__(self,input_size,nb_channels,grn_h_dim,grn_out_dim,contextual_static_dim,dropout,grn = False):
        super(VariableSelectionNetwork,self).__init__()
        self.attn = Attn_weight_POI(input_size*nb_channels,grn_h_dim,nb_channels,contextual_static_dim,dropout)


        if grn :
            self.transformed_inputs = nn.ModuleList([GRN(input_size, grn_h_dim, grn_out_dim,contextual_static_dim=None,dropout=dropout) for i in range(nb_channels)])
        else:
            self.fc1 = nn.Linear(nb_channels,grn_h_dim)
            self.fc2 = nn.Linear(grn_h_dim,grn_out_dim*nb_channels)
            self.relu = nn.ReLU()
        self.grn = grn
    def forward(self,x,x_c=None):
        '''
        Inputs: 
        -------
        x : 3-th order tensor:   [B,C,L]
        x_c : 2-th order tensor:  [B,z_contextual]

        Ouputs: 
        -------
        output : 2-th order tensor:  [B,grn_out_dim] 
        '''

        # Flatten Input: [B,C,L] -> [B,C*L]
        nb_features,dim_input = x.size(1),x.size(2)  # C,L
        flattened_x = x.reshape(x.size(0),-1)

        # GRN on each POIs separately 
        if self.grn:
            out_grn = torch.stack([self.transformed_inputs[k](flattened_x[:,k*dim_input:(k+1)*dim_input]) for k in range(nb_features)],dim=1)
        else:
            # [B,C] -> [B,H] 
            x_out = self.fc1(x[:,:,-1])
            x_out = self.relu(x_out) 
            # [B,H] -> [B,Z*C]
            x_out = self.fc2(x_out)
            # [B,H] -> [B,C,Z]    
            out_grn = x_out.reshape(x_out.size(0),nb_features,-1)

                    # torch.stack(self.transformed_inputs[k](x[Ellipsis, i*self.input_size:(i+1)*self.input_size]) for k in range(self.output_size)], axis=-1)

        # GRN on POIs matrix summed with (repeated) embedding 
        attn_weighs = self.attn(flattened_x,x_c)

        # Aggregation of information from each channels:  
        combined = torch.einsum('bcl,bc->bl',out_grn,attn_weighs)
        return(combined,attn_weighs)

# ============================           END            ============================
# ============================ ======================== ============================


# ============================ ======================== ============================
# ============================ Very Simple Linear Model ============================

class Linear(nn.Module):
    '''
    2 Layer Perceptron with dropout 
    >>> No Gate
    >>> No residual Unit 
    '''
    def __init__(self,input_size,grn_h_dim,grn_out_dim,contextual_static_dim,dropout):
        super(Linear,self).__init__()
        self.input_size = input_size
        self.grn_out_dim = grn_out_dim
        self.dense1 = nn.Linear(self.input_size,grn_h_dim) 
        if contextual_static_dim is not None:
            self.dense_contextual = nn.Linear(contextual_static_dim,grn_h_dim)
        self.dense2 = nn.Linear(grn_h_dim,self.grn_out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,x_c=None):
        '''
        Inputs: 
        -------
        x : 3-th order tensor:  [B,C,input_size] 
        >>> B: Batch-Size
        >>> C: Channel dimension -> Number of POIs * Number of Application 
        >>> input_size : usually sequence length L

        x_c : 2-th order tensor:  [B,z_contextual] 
        >>> z_contextual: dimension of contextual embedding for static information

        Ouputs: 
        -------
        output : 3-th order tensor:  [B,C,grn_out_dim] 
        '''
        #[B,C,L] -> [B,C,h_dim]
        x = self.dense1(x)
        if x_c is not None:
            #[B,z_contextual] -> [B,h_dim] 
            contextual_info = self.dense_contextual(x_c)
            # PAS BON ON VEUT IDEALEMENT QUE CE SOIT DIFFERENT  (same contextual information added to every POIs)
            # [B,C,h_dim] + [B,h_dim] -> [B,C,h_dim]   
            x  = x+contextual_info

        x = self.dense2(x)
        x = self.dropout(x)
        return(x)

class Simple_weight_POI(nn.Module):
    def __init__(self,input_size,grn_h_dim,nb_channels,contextual_static_dim,dropout):
        super(Simple_weight_POI,self).__init__()
        self.linear = Linear(input_size,grn_h_dim,nb_channels,contextual_static_dim,dropout)  # (input_size*nb_channels, grn_h_dim, nb_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x,x_c):
        '''
        Inputs: 
        -------
        x : 3-th order tensor:  [B,C,L]
        x_c : 2-th order tensor:  [B,z_contextual]

        Ouputs: 
        -------
        output : 2-th order tensor:  [B,C] 
        '''
        out_grn = self.linear(x,x_c)
        attn_weight = self.softmax(out_grn)
        return(attn_weight)



class SimpleVariableSelection(nn.Module):
    def __init__(self,input_size,nb_channels,grn_h_dim,grn_out_dim,contextual_static_dim,dropout):
        super(SimpleVariableSelection,self).__init__()
        self.attn = Simple_weight_POI(input_size*nb_channels,grn_h_dim,nb_channels,contextual_static_dim,dropout)
        self.transformed_inputs = nn.ModuleList([Linear(input_size, grn_h_dim, grn_out_dim,contextual_static_dim=None,dropout=dropout) for i in range(nb_channels)])
    def forward(self,x,x_c=None):
        '''
        Inputs: 
        -------
        x : 3-th order tensor:   [B,C,L]
        x_c : 2-th order tensor:  [B,z_contextual]

        Ouputs: 
        -------
        output : 2-th order tensor:  [B,grn_out_dim] 
        '''

        # Flatten Input: [B,C,L] -> [B,C*L]
        nb_features,dim_input = x.size(1),x.size(2)  # C,L
        flattened_x = x.reshape(x.size(0),-1)

        # GRN on each POIs separately 
        out_grn = torch.stack([self.transformed_inputs[k](flattened_x[:,k*dim_input:(k+1)*dim_input]) for k in range(nb_features)],dim=1)
                # torch.stack(self.transformed_inputs[k](x[Ellipsis, i*self.input_size:(i+1)*self.input_size]) for k in range(self.output_size)], axis=-1)

        # GRN on POIs matrix summed with (repeated) embedding 
        attn_weighs = self.attn(flattened_x,x_c)

        # Aggregation of information from each channels:  
        combined = torch.einsum('bcl,bc->bl',out_grn,attn_weighs)
        return(combined,attn_weighs)

# ============================           END            ============================
# ============================ ======================== ============================


# ============================ ======================== ============================
# ============================           Attention Classic pour selectionner les series      ============================

class AttentionPooling(nn.Module):
    def __init__(self, input_length, d_model):
        super(AttentionPooling, self).__init__()
        self.proj = nn.Linear(input_length, d_model)
        self.query = nn.Parameter(torch.randn(d_model))
        nn.init.normal_(self.query, mean=0, std=0.1)

    def forward(self, x,x_c = None):
        # x: [B, P, L]
        
        # Projection des P times-séries dans un espace de dimension d_model
        # embeddings: [B, P, d_model]
        embeddings = self.proj(x)
        
        # Calcul des poids d'attention
        # scores: [B, P]
        scores = torch.matmul(embeddings, self.query)
        scores = scores / math.sqrt(embeddings.size(-1))
        
        # Normalisation softmax pour obtenir les poids
        attn_weights = torch.softmax(scores, dim=1)  # [B, P]
        
        # Agrégation pondérée
        # context: [B, d_model]
        context = torch.sum(embeddings * attn_weights.unsqueeze(-1), dim=1)
        
        return context,attn_weights

# ============================           END            ============================
# ============================ ======================== ============================


# ============================ ======================== ============================
# ============================  Scaled Dot Product     ============================
class ScaledDotProduct(nn.Module):
    def __init__(self, input_length, d_model):
        super(ScaledDotProduct, self).__init__()
        self.proj = nn.Linear(input_length, d_model)

        self.d_model = d_model

        self.W_q = nn.Parameter(torch.cuda.FloatTensor(d_model, d_model)) if torch.cuda.is_available() else nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.W_k = nn.Parameter(torch.cuda.FloatTensor(d_model, d_model)) if torch.cuda.is_available() else nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.W_v = nn.Parameter(torch.cuda.FloatTensor(d_model, d_model)) if torch.cuda.is_available() else nn.Parameter(torch.FloatTensor(d_model, d_model))

        self.softmax = nn.Softmax(dim = -1)


    def forward(self, x,x_c = None):
        # x: [B, P, L]
        
        # Projection des P times-séries dans un espace de dimension d_model
        # embeddings: [B, P, d_model]
        embeddings = self.proj(x)
        
        #[B, P, d_model]
        Q = torch.matmul(embeddings,self.W_q)
        K = torch.matmul(embeddings,self.W_k)
        V = torch.matmul(embeddings,self.W_v)

        #[B, d_model, P]
        K = K.permute(0,2,1)

        #[B, P, P]
        scaled_compat = torch.matmul(Q,K)*1.0/math.sqrt(self.d_model)
        attn_weights = self.softmax(scaled_compat)

        #[B, P, P] x [B, P, d] ->   [B, P, d]
        context = torch.matmul(attn_weights,V)

        #[B, P, d] ->   [B,d]
        context = torch.sum(context, dim=1)
        
        return context,attn_weights





class model(nn.Module):
    def __init__(self,List_input_sizes,List_nb_channels,grn_h_dim,grn_out_dim,contextual_static_dim,dropout):
        super(model,self).__init__()
        # Attention avec Scaled Dot Product pour chaque station:
        if True: 
            self.model = nn.ModuleList([ScaledDotProduct(input_size, grn_out_dim)
                                        for input_size,_ in zip(List_input_sizes,List_nb_channels)])

        # Attention Pooling 'simple':
        if False: 
            self.model = nn.ModuleList([AttentionPooling(input_size, grn_out_dim)
                                        for input_size,_ in zip(List_input_sizes,List_nb_channels)])
        # Gating Mecanisme avec GRN 
        if False: 
            self.model = nn.ModuleList([VariableSelectionNetwork(input_size,nb_channels,grn_h_dim,grn_out_dim,contextual_static_dim,dropout) 
                                        for input_size,nb_channels in zip(List_input_sizes,List_nb_channels)])
            
        # Gating Mecanisme simple avec quelques FC layers: 
        if False: 
            self.model = nn.ModuleList([SimpleVariableSelection(input_size,nb_channels,grn_h_dim,grn_out_dim,contextual_static_dim,dropout) 
                                        for input_size,nb_channels in zip(List_input_sizes,List_nb_channels)])
    def forward(self,List_of_x,x_c=None):
        '''
        Inputs: 
        -------
        List_of_x : list of N 3-th rder torch.Tensor of dimension [B,C_i,L]

        Ouputs: 
        -------
        output : List of N 2-th order torch.Tensor:  [B,grn_out_dim] 
        '''
        # x = List_of_x[k]    Shape: [B,C_i,L]
        # self.model[k](List_of_x[k]) = (combined, attn_weight). We only keep 'combined'.
        return([self.model[k](List_of_x[k],x_c)[0] for k in range(len(List_of_x))])