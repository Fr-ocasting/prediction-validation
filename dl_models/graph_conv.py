from torch import nn
import torch

class graphconv(nn.Module):
    def __init__(self,c_in,c_out,enable_bias,graph_conv_act_func,K = 2):
        super(graphconv,self).__init__()   # Demande a ce qu'on récupère les méthodes de la classe parent :  'nn.module'
        self.c_in = c_in
        self.c_out = c_out
        self.enable_bias = enable_bias
        self.graph_conv_act_func = graph_conv_act_func
        self.K = K

        self.relu = nn.ReLU()

    def forward(self,x,gcnconv_matrix):
        B, C, L, N = x.shape
        n_mat =  gcnconv_matrix.shape[0]

        x = x.reshape(-1, c_in)  #[B, C_in, L, N] -> [BLN, C_in]
        x = torch.einsum('ab, cbd->cad',x,self.weight)   # [BLN,C_in], [K,C_in,C_out] -> [K,BLN,C_out]
        x = x.view(self.K, B*L,N,-1)  #[K,BLN,C_out] ->  [K,BL,N,C_out] 
        x = torch.einsum('ecab,ecbd->ecad',gcnconv_matrix,x)  #[n_adj,BL,N1,N2] ,[K,BL,N2,C_out]  -> [K,BL,N1,C_out] 

        if self.enable_bias:
            x = x + self.bias
        
        x = x.view().permute()
        x = self.relu(x)

        return(x)

