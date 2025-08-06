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

        #[B,C,N,L] -> [B,C,N,h_dim]
        #print('\nx size avant gru: ',x.size())
        x = self.dense1(x)
        #print('x size après dense: ',x.size())
        if x_c is not None:
            #[B,N,z_contextual] -> [B,N,h_dim] 
            #print('calendar data avant passage dans dense: ',x_c.size())
            x_c = self.dense_contextual(x_c)  
            #print('calendar data après passage dans dense: ',x_c.size())
            # PAS BON ON VEUT IDEALEMENT QUE CE SOIT DIFFERENT  (same contextual information added to every POIs)
            # [B,C,h_dim] + [B,h_dim] -> [B,C,h_dim]   
            x  = x+x_c

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
        scaled_compact = torch.matmul(Q,K)*1.0/math.sqrt(self.d_model)

        attn_weights = self.softmax(scaled_compact)

        #[B, P, P] x [B, P, d] ->   [B, P, d]
        context = torch.matmul(attn_weights,V)

        #[B, P, d] ->   [B,d]
        context = torch.sum(context, dim=1)
        
        return context,attn_weights

# ============================ ======================== ============================
# ============================  AttentionGRU ============================
class AttentionGRU(nn.Module):
    def __init__(self, input_length1,input_length2, d_model,grn_h_dim,num_heads,dropout):
        super(AttentionGRU, self).__init__()

        #print('\n>>>>>>>>> input_length1,grn_h_dim,d_model,input_length2 :',input_length1,grn_h_dim,d_model,input_length2)
        self.gru  = GRN(input_length1,grn_h_dim,d_model,input_length2,dropout)
        self.attention = MultiHeadAttention(query_dim=input_length1,key_dim=input_length2, d_model=d_model,num_heads=num_heads,dropout = dropout)

    def forward(self, x_trafic,x_dynamic,x_known):
        ''''
        x_trafic : 2-th order Tensor : historical sequence of trafic flow [B,L]
        x_context : 3-th order Tensor : R Embedding or P_i contextual data associated to the spatial unit of x_trafic [B,P_i,L]
        '''
        # MultiHeadAttention
        #print('x_trafic: ',x_trafic.size())  #[B,L]  -> torch.Size([32,7])  
        #print('x_dynamic: ',x_dynamic.size()) #[B, P, L']  -> torch.Size([32, 4, 7]) 
        #print('x_known None: ',x_known)

        # Gru to design 'query' 
        
        #query = self.gru(x_trafic,x_known)
        query = x_trafic
                                                # query, key, values
        enhanced_x,attn_weights =self.attention(query,x_dynamic,x_dynamic) # self.attention(query,x_dynamic,x_dynamic)
        return enhanced_x,attn_weights
    

class MultiHeadAttention(nn.Module):

    def __init__(self, query_dim, key_dim, d_model,num_heads,dropout,keep_topk=False, proj_query = True):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        assert d_model % num_heads == 0, f"d_model={d_model} must be divisible by num_heads={num_heads}"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        if proj_query:
            self.W_q = nn.Parameter(torch.cuda.FloatTensor(query_dim, d_model)) if torch.cuda.is_available() else nn.Parameter(torch.FloatTensor(query_dim, d_model))
            nn.init.xavier_uniform_(self.W_q)
        else:
            self.W_q = None
            self.repeat_identical_query_heads = True
        self.W_k = nn.Parameter(torch.cuda.FloatTensor(key_dim, d_model)) if torch.cuda.is_available() else nn.Parameter(torch.FloatTensor(key_dim, d_model))
        self.W_v = nn.Parameter(torch.cuda.FloatTensor(key_dim, d_model)) if torch.cuda.is_available() else nn.Parameter(torch.FloatTensor(key_dim, d_model))

        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)



        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)

        # self.layer_norm = nn.LayerNorm(self.d_k)

        # ---- Add normalization for query and key/values before Attention ----
        self.layer_normq = nn.LayerNorm(query_dim)
        self.layer_normkv = nn.LayerNorm(key_dim)
        # ----

        # --- Add linear proj if query_dim != d_model ---
        if query_dim != d_model:
            self.res_proj = nn.Linear(query_dim, d_model)
        else:
            self.res_proj = None
        # ---

        self.keep_topk = keep_topk  # If True then keep only the top 10% of the attention weights
        self.attention_grad_norms = []

    def align_axis(self,query,key,values):
        # Case where only 1 traffic time-serie for P contextual data
        if query.dim() == 2: query = query.unsqueeze(1)
        if key.dim() == 2: key = key.unsqueeze(1)
        if values.dim() == 2: values = values.unsqueeze(1)
        return query,key,values
    
    def padding_sequence_length(self,query,key,values):
        ''' Case where sequence length is not the same for query and key/values 
        Pad 0 at the first dimension and keep the last dimension the same:
        '''
        if query.size(-1) != key.size(-1):
            if query.size(-1) < key.size(-1):
                query = torch.nn.functional.pad(query, (key.size(-1) - query.size(-1),0), mode='constant', value=0)

            else:
                key = torch.nn.functional.pad(key, (query.size(-1) - key.size(-1),0), mode='constant', value=0)
                values = torch.nn.functional.pad(values, (query.size(-1) - values.size(-1),0), mode='constant', value=0)
        return query,key,values

    def split_heads(self,x,is_query=False):
        # print('x size before split_heads: ',x.size())
        # print('is query: ',is_query)
        # print('self.repeat_identical_query_heads: ',self.repeat_identical_query_heads)
        B, P, d_model = x.size()
        # if is_query and self.repeat_identical_query_heads:
        #     x = x.unsqueeze(-1)
        #     x = x.repeat(1,1,1,self.num_heads)
        #     return x.permute(0,3,1,2)
        # else:
        return x.view(B, P, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        # print('x size before combine_heads: ',x.size())
        B, num_heads, P, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(B, P, self.d_model)

    def compute_scaled_dot_product(self,Q,K,V):
        '''
        Compute the scaled dot product with Q,K,V :
        Q : [B,n_heads,nb_units,d_k]    # nb_units = 1 if 'per_station'. Otherwise = len(spatial_units)
        K : [B,n_heads,P,d_k]           # P: numer of nodes, d_k : embedding dimension
        V : [B,n_heads,P,d_k]   


        Output: 
        '''
        K = K.transpose(-2, -1)

        #[B,n_heads, nb_units, d_k]x[B,n_heads, d_k, P] -> [B,n_heads, nb_units, P]
        scaled_compact = torch.matmul(Q,K)*1.0/math.sqrt(self.d_k)

        # Keep only 10% of the best attention weights: 
        if self.keep_topk:
            if scaled_compact.size(-1)>10:
                nb_of_values = 10 #max(10,int(scaled_compact.size(-1)*0.1) )
                _, topk_idx = torch.topk(scaled_compact, nb_of_values, dim=-1) 
                mask = torch.ones_like(scaled_compact) * (-1e6)  # -inf
                mask.scatter_(-1, topk_idx, 0)                           # add 0 on the top k 
                scaled_compact = scaled_compact + mask                    # on garde, le reste = 0
                attn_weights = self.softmax(scaled_compact)  
            else:
                attn_weights = self.softmax(scaled_compact)  
                self.dropout(attn_weights)
        else:
            attn_weights = self.softmax(scaled_compact)  
            self.dropout(attn_weights)

        # Hook to keep track on attention weight grad:
        self.attn_weight_for_hook = attn_weights
        if self.attn_weight_for_hook.requires_grad:
            self.attn_weight_for_hook.register_hook(self._save_grad_norm)


        #print('scaled_compact.size: ',scaled_compact.size())
        #print('attn_weights: ',attn_weights.size())
        #print('attn_weights sum on each head: ',[attn_weights[:,h,nb_unit,:].sum(-1)[0] for h in range(Q.size(1)) for nb_unit in range(Q.size(2))])

        #[B,n_heads, nb_units, P] x [B,n_heads, P, d] ->   [B,n_heads, nb_units, d]
        context = torch.matmul(attn_weights,V)
        return context,attn_weights

    def _save_grad_norm(self, grad):
        """Hook"""
        if grad is not None:
            self.attention_grad_norms.append(grad.norm().item())

    def forward(self, query,key,values):
        '''
        query: From x_traffic    -> [B,nb_units,L] or [B,L]
        key : From x_dynamic    -> [B,P,L]    --|
        values: From x_dynamic    -> [B,P,L]  --|---> Same object
        '''
        original_query = query
        batch_size = key.size(0)
        # print('\nquery,key,values before align: ',query.size(),key.size(),values.size())
        query,key,values = self.align_axis(query,key,values)
        # print('query,key,values after align: ',query.size(),key.size(),values.size())
        # print('projection matrix Wq,Wk,Wv: ',self.W_q.size(),self.W_k.size(),self.W_v.size())
        # print('query values: ',query[0,0,:])
        # print('key values: ',key[0,0,:])



        #Projection to a laten space of dimenison d: [B,n_heads, P, d_k]
        #   query x    self.W_q   -->     QxW_q     
        # [B,1,L] x [B,L,d_model] --> [B,1,d_model] --split_head--> [B,n_heads,1,d_model//n_heads] 
        # print('query,key,values before proj: ',query.size(),key.size(),values.size())
        #print('W_q,W_k,W_v: ',self.W_q.size(),self.W_k.size(),self.W_v.size())
        if self.W_q is not None:
            query,key,values = self.padding_sequence_length(query,key,values)
            # Q = self.layer_norm(self.split_heads(torch.matmul(query,self.W_q)))
            # Q = self.split_heads(torch.matmul(query,self.W_q))
            Q = self.split_heads(torch.matmul(self.layer_normq(query),self.W_q))
            
        else:
            # Q = self.layer_norm(self.split_heads(query,is_query=True))
            # Q = self.split_heads(query,is_query=True)
            Q = self.split_heads(self.layer_normq(query),is_query=True)

    
        # K = self.split_heads(torch.matmul(key,self.W_k))
        K = self.split_heads(torch.matmul(self.layer_normkv(key),self.W_k))
        # K = self.layer_norm(self.split_heads(torch.matmul(key,self.W_k)))   

        # V = self.split_heads(torch.matmul(values,self.W_v))
        V = self.split_heads(torch.matmul(self.layer_normkv(values),self.W_v))
        # V = self.layer_norm(self.split_heads(torch.matmul(values,self.W_v)))


        # print('Q,K,V after proj: ',Q.size(),K.size(),V.size())
        context,attn_weights = self.compute_scaled_dot_product(Q,K,V)
        # print('context,attn_weights: ',context.size(),attn_weights.size())

        #[B,n_heads, nb_units, d_k] -> [B, nb_units, d_models]     
        context = self.combine_heads(context)
        # print('context after combine_heads: ',context.size())
        combined_Q = self.combine_heads(Q)

        #[B, nb_units, d_models] -> [B,d_models] if 'per_station'. Otherwise  do nothing
        #context = torch.sum(context, dim=-2)
        context = context.squeeze()
        # tackle case where squeeze if B = 1
        if batch_size==1:
            context = context.unsqueeze(0)

        # --- Residual Layer
        if self.res_proj is not None:
            residual = self.res_proj(original_query)
        else:
            residual = original_query
        
        context_output = residual + self.dropout(context)
        # ----
        

        return combined_Q,context_output,attn_weights


class model(nn.Module):
    def __init__(self,List_input_sizes,List_nb_channels,grn_h_dim,grn_out_dim,contextual_static_dim,dropout,x_input_size,num_heads = None):
        super(model,self).__init__()
        # Attention avec Scaled Dot Product pour chaque station:
        self.model = nn.Identity()
        '''Dernier en date fonctionnel '''
        if False:
            if type(List_input_sizes) == list:
                self.model = nn.ModuleList([AttentionGRU(x_input_size,input_size, grn_out_dim,grn_h_dim,num_heads,dropout)
                                            for input_size in List_input_sizes])      
                self.mini_att_models = True
            else:
                self.model = AttentionGRU(x_input_size,List_input_sizes, grn_out_dim,grn_h_dim,num_heads,dropout)      
                self.mini_att_models = False
        if False: 
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
            
    def forward(self,x,List_of_x,x_c=None):
        '''
        Inputs: 
        -------
        x : batch of sequence of historical trafic flow  [B,N,L]
        List_of_x : list of N 3-th rder torch.Tensor of dimension: N*[B,C_i,L]
        x_c : contextual statique data

        Ouputs: 
        -------
        output : List of N 2-th order torch.Tensor:  [B,grn_out_dim] 
        '''
        # x = List_of_x[k]    Shape: [B,C_i,L]
        # self.model[k](List_of_x[k]) = (combined, attn_weight). We only keep 'combined'.

        ''' METTRE TRUE '''
        if False:
            if self.mini_att_models:
                return([self.model[k](x[:,:,k,:].squeeze(),List_of_x[k],x_c)[0] for k in range(len(List_of_x))])
            else:
                # List_of_x is here actually only a single tensor
                return(self.model(x.squeeze(),List_of_x,x_c)[0])
        return List_of_x


