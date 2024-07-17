import torch
import torch.nn as nn

from dl_models.vision_models.conv_3D import conv3x3x1,conv1x1x3,conv1x1x1

class trivial_block_2PLus1D(nn.Module):
    def __init__(self,c_in,out_dim,activation = True):
        super(trivial_block_2PLus1D,self).__init__()

        n_3d_parameters1 = (c_in * out_dim * 3 * 3 * 3)
        n_2p1d_parameters1 = (c_in * 3 * 3) + (3 * out_dim)
        h_dim = n_3d_parameters1 // n_2p1d_parameters1

        self.conv1_s = conv3x3x1(c_in, h_dim)
        self.bn1_s = nn.BatchNorm3d(h_dim)
        self.conv1_t = conv1x1x3(h_dim, out_dim)
        self.bn1_t = nn.BatchNorm3d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.activation = activation

    def forward(self,x):
        '''
        Inputs :
        --------
        5-th order tensor [B,c_in,H,W,L]

        Outputs:
        --------
        5-th order tensor [B,out_dim,H,W,L]
        '''
        x = self.conv1_s(x)
        x = self.bn1_s(x)
        x = self.relu(x)
        x = self.conv1_t(x)
        x = self.bn1_t(x)
        if self.activation :
            x = self.relu(x)  
        return x



class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, downsample=None):
        super().__init__()
        self.trivial_bloc1 = trivial_block_2PLus1D(in_planes,planes,activation = True)
        self.trivial_bloc2 = trivial_block_2PLus1D(planes,planes,activation = False)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample

    def forward(self, x):
        '''
        Inputs :
        --------
        5-th order tensor [B,in_planes,H,W,L]

        Outputs:
        --------
        5-th order tensor [B,planes,H,W,L]
        '''

        residual = x

        out = self.trivial_bloc1(x)
        out = self.trivial_bloc2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_2_1D_block(nn.Module):
    def __init__(self,in_planes,planes):
        super(ResNet_2_1D_block,self).__init__()
        downsample = nn.Sequential(conv1x1x1(in_planes, planes),
                                nn.BatchNorm3d(planes)
                                )
        self.block = BasicBlock(in_planes, planes, downsample=downsample)

    def forward(self,x):
        '''
        Inputs :
        --------
        5-th order tensor [B,in_planes,H,W,L]

        Outputs:
        --------
        5-th order tensor [B,planes,H,W,L]
        '''
        return self.block(x)
    

if __name__ == '__main__':
    from utilities_DL import forward_and_display_info

    B,N,C,H,W,L = 32, 40, 4, 22,22,6
    netmob = torch.randn(B,N,C,H,W,L)
    netmob_station_i = netmob[:,0,:,:,:,:]

    in_planes = C
    planes = 64 # 128 # 256 # 512

    block =  ResNet_2_1D_block(in_planes, planes)
    output = forward_and_display_info(block,netmob_station_i)

    trivial_bloc = trivial_block_2PLus1D(in_planes,planes)
    output = forward_and_display_info(trivial_bloc,netmob_station_i)
