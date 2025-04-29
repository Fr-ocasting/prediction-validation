"""
Copy from https://github.com/guoshnBJTU/ASTGCN-2019-pytorch/blob/master/train_ASTGCN_r.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal import:
from dl_models.ASTGCN.lib.utils import scaled_Laplacian, cheb_polynomial


class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))

        self.init_parameters()
        
    def init_parameters(self):
        # Uniform distribution on 1D and Xavier uniform distribution on tensors >= 2D
        nn.init.uniform_(self.W1)   # 1D 
        nn.init.xavier_uniform_(self.W2)
        nn.init.uniform_(self.W3)
        nn.init.xavier_uniform_(self.bs)
        nn.init.xavier_uniform_(self.Vs)


    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''
        # print('\nStart Spatial Attention Layer: ')
        # print('nan in x, W1, W2: ',torch.isnan(x).any().item(),torch.isnan(self.W1).any().item(),torch.isnan(self.W2).any().item())
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)
        # print('nan in lhs: ',torch.isnan(lhs).any().item())
        # print('nan in rhs: ',torch.isnan(rhs).any().item())
        # print('nan in product: ',torch.isnan(product).any().item())
        # print('nan in S: ',torch.isnan(S).any().item())
        # print('nan in S_normalized: ',torch.isnan(S_normalized).any().item())

        return S_normalized


class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

        self.init_parameters()

    def init_parameters(self):
        # Uniform distribution on 1D and Xavier uniform distribution on tensors >= 2D
        for theta in self.Theta:
            nn.init.xavier_uniform_(theta)

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                T_k_with_at = T_k.mul(spatial_attention)   # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.init_parameters()
        
    def init_parameters(self):
        # Uniform distribution on 1D and Xavier uniform distribution on tensors >= 2D
        nn.init.uniform_(self.U1)   # 1D 
        nn.init.xavier_uniform_(self.U2)
        nn.init.uniform_(self.U3)
        nn.init.xavier_uniform_(self.be)
        nn.init.xavier_uniform_(self.Ve)

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        #print('\nStart Temporal Attention Layer: ')
        #print('nan in x, U1, U2: ',torch.isnan(x).any().item(),torch.isnan(self.U1).any().item(),torch.isnan(self.U2).any().item())
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        #print('\nnan in lhs: ',torch.isnan(lhs).any().item())
        #print('nan in rhs: ',torch.isnan(rhs).any().item())
        #print('nan in product: ',torch.isnan(product).any().item())
        #print('nan in E: ',torch.isnan(E).any().item())
        #print('nan in E_normalized: ',torch.isnan(E_normalized).any().item())

        return E_normalized


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])
        self.init_parameters()

    def init_parameters(self):
        # Uniform distribution on 1D and Xavier uniform distribution on tensors >= 2D
        for theta in self.Theta:
            nn.init.xavier_uniform_(theta)

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class ASTGCN_block(nn.Module):
    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_of_vertices, num_of_timesteps):
        super(ASTGCN_block, self).__init__()
        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)  #需要将channel放到最后一个维度上

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        #print('\nx start block_i:')
        #print('nan in x: ',torch.isnan(x).any().item())
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # TAt
        temporal_At = self.TAt(x)  # (b, T, T)
        #print('nan in x after TAt: ',torch.isnan(temporal_At).any().item())

        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        #print('nan in x after matmul reshape: ',torch.isnan(x_TAt).any().item())
        # SAt
        spatial_At = self.SAt(x_TAt)
        #print('nan in x after SAt : ',torch.isnan(spatial_At).any().item())

        # cheb gcn
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)  # (b,N,F,T)
        #print('nan in x after cheb_conv_SAt : ',torch.isnan(spatial_gcn).any().item())
        # spatial_gcn = self.cheb_conv(x)

        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,3)的卷积核去做->(b,F,N,T)
        #print('nan in x after time_conv : ',torch.isnan(time_conv_output).any().item())

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)
        #print('nan in x after residual_conv : ',torch.isnan(x_residual).any().item())

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        #print('nan in x after ln(F(ReLU)) : ',torch.isnan(x_residual).any().item())
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)



        return x_residual


class ASTGCN(nn.Module):

    def __init__(self, device, nb_block, C, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, step_ahead, L, n_vertex):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''
        super(ASTGCN, self).__init__()
        # Apply correspondance between our syntaxe and the one in the paper:
        self.DEVICE = device
        self.in_channels = C
        self.len_input = L
        self.num_for_predict = step_ahead
        self.num_of_vertices = n_vertex


        self.BlockList = nn.ModuleList([ASTGCN_block(self.DEVICE, self.in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, self.num_of_vertices, self.len_input)])

        self.BlockList.extend([ASTGCN_block(self.DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter, 1, cheb_polynomials, self.num_of_vertices, self.len_input//time_strides) for _ in range(nb_block-1)])

        self.final_conv = nn.Conv2d(int(self.len_input/time_strides), self.num_for_predict, kernel_size=(1, nb_time_filter))


        self.to(self.DEVICE)

    def forward(self, x,extracted_feature=None,time_elt=None):
        '''
        :param x: (B, F_in, N_nodes, T_in)  but PERMUTE to be (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        #print('\nx entry: ')
        #print('nan in x: ',torch.isnan(x).any())
        x = x.permute(0,2,1,3)  # (B, F_in, N_nodes, T_in) -> (B, N_nodes, F_in, T_in) 
        for block in self.BlockList:
            x = block(x)
        #print('\nx after blocks: ')
        #print('nan in x: ',torch.isnan(x).any())

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        #print('\nx after final_conv: ')
        #print('nan in x: ',torch.isnan(x).any())
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        return output


def make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx, num_for_predict, len_input, num_of_vertices):
    '''

    :param DEVICE:
    :param nb_block:
    :param in_channels:
    :param K:
    :param nb_chev_filter:
    :param nb_time_filter:
    :param time_strides:
    :param cheb_polynomials:
    :param nb_predict_step:
    :param len_input
    :return:
    '''
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
    model = ASTGCN(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model