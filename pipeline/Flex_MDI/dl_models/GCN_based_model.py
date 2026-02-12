import torch.nn as nn
import torch 

# ====================
# Graph Convolution 
# ====================
class graphconv(nn.Module):
    def __init__(self,c_in,c_out,graph_conv_act_func,K = 2,enable_bias =True):
        super(graphconv,self).__init__()   # Demande a ce qu'on récupère les méthodes de la classe parent :  'nn.module'
        self.c_in = c_in
        self.c_out = c_out
        self.enable_bias = enable_bias
        self.graph_conv_act_func = graph_conv_act_func
        self.K = K
        if torch.cuda.is_available():
            self.weight = nn.Parameter(torch.cuda.FloatTensor(K,c_in,c_out))   # Initialize with wierd weight like 0e-30 or 1e38. Might not be totally adaptated ...
        else :
            self.weight = nn.Parameter(torch.FloatTensor(K,c_in,c_out))
        self.relu = nn.ReLU()

    def forward(self,x,gcnconv_matrix):
        B, C, L, N = x.shape
        n_mat =  gcnconv_matrix.shape[0]

        x = x.reshape(-1, self.c_in)  #[B, C_in, L, N] -> [BLN, C_in]
        x = torch.einsum('ab, cbd->cad',x,self.weight)   # [BLN,C_in], [K,C_in,C_out] -> [K,BLN,C_out]
        x = x.view(self.K, B*L,N,-1)  #[K,BLN,C_out] ->  [K,BL,N,C_out] 
        x = torch.einsum('ecab,ecbd->ecad',gcnconv_matrix,x)  #[n_adj,BL,N1,N2] ,[K,BL,N2,C_out]  -> [K,BL,N1,C_out] 

        if self.enable_bias:
            x = x + self.bias
        
        x = x.view().permute()
        x = self.relu(x)

        return(x)
    
# ====================
# MULTI Graph Convolution 
# ====================
class MGCN_conv(nn.Module):
    def __init__(self,c_in,c_out,n_adj=2,enable_bias =True):
        super(MGCN_conv,self).__init__()
        self.n_adj = n_adj    
        self.c_in = c_in
        self.c_out = c_out
        self.weight = torch.FloatTensor(n_adj,c_in,c_out)  #[C_in,C_out]   ou pour Einsum: [l,h]
        self.bias = torch.FloatTensor(c_out)
        self.relu = nn.ReLU()

    def forward(self,x,adj_matrix):
        # x.shape = [B,N2,L]
        B,N2,L = x.shape
        # A.shape = [n_adj,N1,N2]
        n_adj,N1,N2 = adj_matrix.shape

        #adj_matrix = torch.unsqueeze(adj_matrix,0) #Créer un nouvel axe 
        stacked_adj = adj_matrix.repeat(B,1,1) # Concat tout les "samples" (B batch, L tmeporal dimension) long du nouvel axe
        stacked_adj = stacked_adj.reshape(-1,N1,N2)  # Stack les n_adj adjacency matrix B*L fois     shape: [n_adj*B, N1,N2] et pour einsum : [b,p,n]

        embedding = torch.einsum('bnl,klh -> bknh',x,self.weight)   # Embedding sur les feature de chaque Noeud
        reshaped_embedding = embedding.reshape(-1,N2,self.c_out)    #Stack la dimension k sur l'axe 0  [n_adj*B,N2,C_out]  et pour einsum : [b,n,h]

        convoluted = torch.einsum('bpn,bnh -> bph',stacked_adj,reshaped_embedding) 
        convoluted = convoluted.reshape(self.n_adj,B,N1,self.c_out)      #Unstack la dimension 0, pour séparer les n_adj matrices d'adjacences

        convoluted = self.relu(convoluted + self.bias)

        return(convoluted,embedding,reshaped_embedding)



if __name__ == '__main__':
    if False : 
        # Test : 
        N1,N2 = 3,3
        adj_matrix = torch.stack([torch.diag(torch.ones(N1)),torch.randn(N1,N2)],dim = 0)  # Matrice d'adjacence identité, personne n'est connecté avec personne
        n_adj = adj_matrix.shape[0]

        B = 4
        L = 6
        x = torch.randn(B,N1,L)

        c_out = 16
        gcn_model = MGCN_conv(L,c_out,n_adj)
        AXW,embedding,reshaped_embedding = gcn_model(x,adj_matrix)

        print("L'embedding a d'abord été opéré sur la dernière dimension (X*W, temporelle), Puis la convolution (A*(XW)) a sommmé les embedding de chacun des voisins (ou Noeud en lien avec le noeud tagret).")
        print(f'X.shape: {x.shape}, AXW.shape:{AXW.shape}')

    if False : 
        x = torch.randn(2,1,3,4)
        A_indep = torch.ones(3,3)

        C_in = x.shape[1]
        C_out = 8
        weight = torch.FloatTensor(C_in,C_out)
        weight = torch.Tensor([[1,2,3,4,5,6,7,8]])

        adj_matrix =A_indep
        # x.shape = [B,C,N2,L]
        B,C_in,N2,L = x.shape
        # A.shape = [n_adj,N1,N2]
        N1,N2 = adj_matrix.shape
        #A.shape = [n_adj,B,N1,N2]
        stacked_adj = adj_matrix.repeat(1,B,1,1)


        embedding = torch.einsum('bcnl,ch -> bhnl',x,weight)

    if False : 
        T = 1000  # Number of available time-slot
        N = 50   # Number of spatial unities
        L = 6   # Historical Length 
        X =torch.randn(T,N,L)
        x = X.unsqueeze(1) #add the channel dimension (here, only "flow')
        print(f'X shape: {x.shape}')
        x_b = x[:32]
        print(f'x_b shape: {x_b.shape}')

        GCN = graphconv(c_in = x.shape[1], c_out = 64, K=2, graph_conv_act_func = 'relu',enable_bias=True)  # K =2 dans MRGNN car considère Pattern et Adj matrix
        B, C, N, L = x_b.shape
        K = 1
        c_out = 64
        n_mat =  A_indep.shape[0]

        print(f'B,C,N,L : [{B},{C},{N},{L}]')