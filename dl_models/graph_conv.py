from torch import nn
import torch
import torch.nn.init as init


class AttnAgg(nn.Module):
    def __init__(self,c_out):
        super(AttnAgg,self).__init__()
        self.coeff = nn.Linear(c_out,1)
        self.softmax = nn.Softmax(-1)

    def forward(self,Multi_x_conv):  # Multi_x_conv : [b,k,n,c_out]
        # Attention coefficient
        attn_coeff = self.coeff(Multi_x_conv)  # Associe un coefficient pour chacun des C_out tensor [b,k,n]
        attn_coeff = torch.einsum('...knh -> ...nhk',attn_coeff) # permute pour avoir [embedding_dim, n_adj] en dernières dim    (ici embedding dim = 1, car un seul coeff attention)
        attn_coeff = self.softmax(attn_coeff)  # puis softmax sur la dernière dim (n_adj)  -> 1 si dim = 1, sinon softmax (i.e moyenne logits)

        # Permute Multi_x_conv
        permuted_Multi_x_conv = torch.einsum('...knl -> ...nkl',Multi_x_conv) # permute pour avoir [n_adj,C_out] en dernière positions

        agg_multi_conv_x = torch.einsum('...nhk,...nkl -> ...nhl',attn_coeff,permuted_Multi_x_conv) # Moyenne des alpha_i x_i sur l'ensemble des n_adj
        agg_multi_conv_x = torch.einsum('...nhl -> ...nl',agg_multi_conv_x)  #Réduit la dimension (avec un softmax sur la dernière dim, retourne 1 pour tout le monde si on a qu'une seule matrice d'adjacence)
        
        return(agg_multi_conv_x)

class Multigraphconv(nn.Module):
    def __init__(self,c_in,c_out,n_adj = 1,gcn_activation = 'relu', init_weight = 'kaiming_uniform' ):
        super(Multigraphconv,self).__init__()  
        self.c_in = c_in
        self.c_out = c_out
        self.gcn_activation = gcn_activation
        self.n_adj = n_adj
        self.relu = nn.ReLU()

        # Weight
        if torch.cuda.is_available():
            self.weight = nn.Parameter(torch.cuda.FloatTensor(n_adj,c_in,c_out))  #[C_in,C_out]   ou pour Einsum: [l,h]
            self.bias = nn.Parameter(torch.cuda.FloatTensor(c_out))
        else:
            self.weight = nn.Parameter(torch.FloatTensor(n_adj,c_in,c_out))  #[C_in,C_out]   ou pour Einsum: [l,h]
            self.bias = nn.Parameter(torch.FloatTensor(c_out))

        self.init_weight = init_weight
        self.init_parameters()

        # Relation d'aggregation (Multi-Graph -> Simple)
        self.agg = AttnAgg(c_out)

    def init_parameters(self):
        if self.init_weight == 'kaiming_uniform':
            init.kaiming_uniform_(self.weight)
        if self.init_weight == 'xavier_uniform':
            init.xavier_uniform_(tensor = self.weight, gain=init.calculate_gain(self.gcn_activation))
        init.zeros_(self.bias)

    

    def forward(self,x,adj_matrix):
        # bnl,klh -> bknh -> Pour chaque couple (b,k), on renvoit la multiplication matricielle nl*lh
        # soit :  [ 1.,  2.,  3.]           [13., 14., 15.]      [102., 108., 114.]
        #         [ 4.,  5.,  6.]      *    [16., 17., 18.] =    [246., 261., 276.]
        #         [ 7.,  8.,  9.]           [19., 20., 21.]      [390., 414., 438.]
        #         [ 10., 11., 12.,]                              [534., 567., 600.]

        # Ne prend que des x déja reshape, avec en deux dernière dimension NxC (Noeuds x Channel sur lequel faire l'embedding et convolution)
        embedded_x = torch.einsum('bnc, kch->bknh',x,self.weight)   # Embedding 
        stacked_adj_matrix = adj_matrix.unsqueeze(0).repeat(x.shape[0],1,1,1)  # Stack b fois la matrice (b = B*L ou B*C, avec L longueur sequence temporelle, C longueur channel )
        Multi_x_conv = torch.einsum('bkpn,bknh->bkph',stacked_adj_matrix,embedded_x)  # Convolution
        x_conv = self.relu(Multi_x_conv + self.bias)  # Relu + Add bias 

        x_agg_conv = self.agg(x_conv)    # Aggregation  des n_adj convolution  

        return(x_agg_conv)

