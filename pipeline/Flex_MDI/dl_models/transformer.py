import torch
import torch.nn as nn 


class MultiHeadAttention(nn.Module):
    def __init__(self,n_head,embed_dim):
        super(MultiHeadAttention,self).__init__()
        self.single_head_dim = embed_dim//n_head
        self.n_head = n_head
        self.embed_dim = embed_dim

        self.W_q = nn.Linear(embed_dim,embed_dim)
        self.W_k = nn.Linear(embed_dim,embed_dim)
        self.W_v = nn.Linear(embed_dim,embed_dim)

        self.softmax = nn.Softmax(dim=-1)

    def scaled_dot_product_attn(self,Q,K,V,mask = None):
        '''
        Q: [B,C,h,N,d]
        K: [B,C,h,N,d]
        V: [B,C,h,N,d]
        '''
        # Attn coefficient: 
        attn_scores = torch.matmul(Q,K.transpose(-2,-1))/self.single_head_dim**0.5
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0,1e-9)
        attn_probs = self.softmax(attn_scores)

        # Context vect:
        context_vect = torch.matmul(attn_probs,V)
        return(context_vect)
    
    def split_heads(self,x):
        B,C,N,L = x.size()
        # x -> [B,C,n_head,N,single_head_dim]
        return(x.view(B,C,N,self.n_head,self.single_head_dim).transpose(2,3))

    def stack_heads(self,x):
        B,C,_,N,d = x.size() 

        # x: [B,C,h,N,d] -> [B,C,N,h,d]
        x = x.transpose(2,3).contiguous()  # allow to keep track on 'order' of the data in memory 

        # x: [B,C,N,h,d]
        x = x.view(B,C,N,-1)

        return(x)

    def forward(self,x1,x2 = None, mask = None):
        if x2 is None:
            x2 = x1
        
        # x1: [B,C,N1,d], x2: [B,C,N2,d] 
        Q = self.split_heads(self.W_q(x1))
        K = self.split_heads(self.W_k(x2))
        V = self.split_heads(self.W_v(x2))

        # context_vect: [B,C,h,N1,d_v] -> [B,C,N1,d]
        contex_vect = self.scaled_dot_product_attn(Q,K,V,mask)
        contex_vect = self.stack_heads(contex_vect)

        return contex_vect



class FeedForward(nn.Module):
    def __init__(self,embed_dim,d_ff,c_out = None,dropout = 0.0):
        super(FeedForward,self).__init__()
        if c_out is None:
            c_out = embed_dim
        self.c_out = c_out
        self.d_ff = d_ff
        self.embed_dim = embed_dim
        self.dense1 = nn.Linear(embed_dim,d_ff)
        self.dense2 = nn.Linear(d_ff,c_out)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.dropout(self.relu(self.dense1(x)))
        x=  self.dense2(x)
        return(x)

class Encoder_layer(nn.Module):
    def __init__(self,embed_dim,d_ff,n_head,dropout = 0.0):
        super(Encoder_layer,self).__init__()
        assert embed_dim%n_head == 0,'Embedding dim must be equal to 0 modulo the number of heads'
        self.d_ff = d_ff
        self.n_head = n_head

        # mha layer 
        self.mha = MultiHeadAttention(n_head,embed_dim)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)

        # output layer : 
        #self.dense = nn.Linear(d_v,d_ff)
        self.feedforward = FeedForward(embed_dim,d_ff,dropout=dropout)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)

        # regularization : 
        self.dropout = nn.Dropout(dropout)

    def forward(self,x1): 
        # dropout + mha    
        x_attn = self.dropout(self.mha(x1))

        # add Residual and then pass through Layer Norm
        x_attn = self.attn_layer_norm(x1 + x_attn)

        # dropout on FF module + add the last output, and then normalize : 

        output = self.ff_layer_norm(x_attn + self.dropout(self.feedforward(x_attn)))

        return(output)

         
class ENCODER(nn.Module):
    ''' 
    Input : x [B,C,N,d] and the output of the 'scaled-dot-product Attention' will have the same dimenions.
    There are 'n_head' head, the output of each of them will be concatenated. Which mean x -> [B,C,N,d*n_head] 
    args** 
    d_qk : dimension of embedding for Query and Key 
    d_v : dimension of embedding for Values. Which will be the last one
    '''
    def __init__(self,input_dim,embed_dim,d_ff,n_head,num_layers,dropout = 0.0):
        super(ENCODER,self).__init__()
        self.d_ff = d_ff
        self.n_head = n_head
        self.num_layers = num_layers
        self.embedding = nn.Linear(input_dim,embed_dim)
        self.encoder_layers = nn.ModuleList([Encoder_layer(embed_dim,d_ff,n_head,dropout) for l in range(num_layers)])

    def forward(self,x1,x2 = None):
        x1 = self.embedding(x1)
        for layer in self.encoder_layers:
            x1 = layer(x1)

        return(x1)
    

class EncoderTransformer(nn.Module):
    def __init__(self,output_dim,input_dim,embed_dim,d_ff,n_head,num_layers,dropout=0.0):
        super(EncoderTransformer,self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.d_ff = d_ff
        self.n_head = n_head
        self.num_layers = num_layers
        self.dropout = dropout

        self.encoder = ENCODER(input_dim,embed_dim,d_ff,n_head,num_layers,dropout)
        self.output_module = FeedForward(embed_dim,embed_dim//2,c_out = 1,dropout = 0.0)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        if len(x.size()) == 3:
            x = x.unsqueeze(1)
        x = self.output_module(self.relu(self.encoder(x)))
        return(x)


if False : 

    class CrossAttention(nn.Module):
        ''' Fait maison, à prendre avec des pincettes. nn.Linear marche évidement au lieu de nn.Parameter.
        On a parfois même de meilleurs résultats car l'initialisation est différente.

        - torch.rand initialise avec une gaussienne de grande variance (std = 1)
        - nn.Linear initialize suivant une 'scaled uniform distribution'
        - On peut donc penser à changer l'initialisation avec une ligne du type :  nn.init.xavier_normal_(self.Wq)

        '''
        def __init__(self,single_head_dim, d_qk, d_v):
            super(CrossAttention,self).__init__()
            self.single_head_dim = single_head_dim
            self.d_qk = d_qk
            self.d_v = d_v

            self.Wq = nn.Parameter(torch.FloatTensor(single_head_dim,d_qk))
            self.Wk = nn.Parameter(torch.FloatTensor(single_head_dim,d_qk))
            self.Wv = nn.Parameter(torch.FloatTensor(single_head_dim,d_v))

            self.softmax = nn.Softmax(-1)

        def forward(self,x1,x2 = None):
            if x2 is None:
                x2 = x1
            Q = torch.matmul(x1,self.Wq) #torch.bmm(x1,self.Wq)  # [B,C,N,d_qk]
            K = torch.matmul(x2,self.Wk) #torch.bmm(x2,self.Wk)  # [B,C,N,d_qk]
            V = torch.matmul(x2,self.Wv) #torch.bmm(x2,self.Wv)  # [B,C,N,d_v]

            attn_coeff = self.softmax(torch.einsum('bcnq,bcqp->bcnp', Q, K.transpose(-2,-1))/(self.d_qk**0.5))
            context_vect = torch.einsum('bcpn,bcnv->bcpv',attn_coeff,V)  #torch.bmm(self.softmax(torch.bmm(Q,K.transpose(-2,-1))/(self.d_qk**0.5)),V)

            return(context_vect)
        
    class MultiHeadAttention(nn.Module):
        def __init__(self,n_head,embed_dim, d_qk, d_v):
            super(MultiHeadAttention,self).__init__()
            self.single_head_dim = embed_dim//n_head
            self.n_head = n_head
            self.heads = nn.ModuleList([CrossAttention(self.single_head_dim,d_qk,d_v) for _ in range(n_head)])

        def forward(self,x1,x2 = None):
            # Split heads 
            B,C,N,_ = x1.shape   # Last dim  = Embedding dim
            x1 = x1.view(B,C,N,self.n_head,self.single_head_dim)

            # Pass through each head :
            torch.cat([h(x1,x2) for h in self.heads],dim = -1)
            # Concat 
            return  

