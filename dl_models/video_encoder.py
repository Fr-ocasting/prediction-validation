import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
from functools import partial

class VideoEncoder_module(nn.Module):
    def __init__(self):
        super(VideoEncoder_module,self).__init__()
        self.blabla = blabla

    def forward(self,x,netmob_video_batch):
        '''
        args 
        -----
        netmob_video_batch : shape [B,N,C,H,W,L] 
        B : batch-size
        N : number of spatial-units
        C : number of channel (i.e number of mobile-phone apps)
        H,W : height and width of image around the spatial unit
        L : length of historical sequence (t-w,t-d,t-6,t-5,t-4,t-3,t-2,t-1)
        '''
        blabla


def conv1x3x3(in_planes, mid_planes, stride=1):
    return nn.Conv3d(in_planes,
                     mid_planes,
                     kernel_size=(1, 3, 3),
                     stride=(1, stride, stride),
                     padding=(0, 1, 1),
                     bias=False)


def conv3x1x1(mid_planes, planes, stride=1):
    return nn.Conv3d(mid_planes,
                     planes,
                     kernel_size=(3, 1, 1),
                     stride=(stride, 1, 1),
                     padding=(1, 0, 0),
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        n_3d_parameters1 = in_planes * planes * 3 * 3 * 3
        n_2p1d_parameters1 = in_planes * 3 * 3 + 3 * planes
        mid_planes1 = n_3d_parameters1 // n_2p1d_parameters1
        self.conv1_s = conv1x3x3(in_planes, mid_planes1, stride)
        self.bn1_s = nn.BatchNorm3d(mid_planes1)
        self.conv1_t = conv3x1x1(mid_planes1, planes, stride)
        self.bn1_t = nn.BatchNorm3d(planes)

        n_3d_parameters2 = planes * planes * 3 * 3 * 3
        n_2p1d_parameters2 = planes * 3 * 3 + 3 * planes
        mid_planes2 = n_3d_parameters2 // n_2p1d_parameters2
        self.conv2_s = conv1x3x3(planes, mid_planes2)
        self.bn2_s = nn.BatchNorm3d(mid_planes2)
        self.conv2_t = conv3x1x1(mid_planes2, planes)
        self.bn2_t = nn.BatchNorm3d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.conv1_s(x)
        out = self.bn1_s(out)
        out = self.relu(out)
        out = self.conv1_t(out)
        out = self.bn1_t(out)
        out = self.relu(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        n_3d_parameters = 3 * self.in_planes * conv1_t_size * 7 * 7
        n_2p1d_parameters = 3 * 7 * 7 + conv1_t_size * self.in_planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        self.conv1_s = nn.Conv3d(n_input_channels,
                                 mid_planes,
                                 kernel_size=(1, 7, 7),
                                 stride=(1, 2, 2),
                                 padding=(0, 3, 3),
                                 bias=False)
        self.bn1_s = nn.BatchNorm3d(mid_planes)
        self.conv1_t = nn.Conv3d(mid_planes,
                                 self.in_planes,
                                 kernel_size=(conv1_t_size, 1, 1),
                                 stride=(conv1_t_stride, 1, 1),
                                 padding=(conv1_t_size // 2, 0, 0),
                                 bias=False)
        self.bn1_t = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        
        # New
        self.layers = nn.ModuleList([self._make_layer(block,
                                       block_inplanes[k],
                                       layers[k],
                                       shortcut_type,
                                       stride=2) for k in range(1,len(block_inplanes))])
        # ...
        '''
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        '''

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[-1] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_s(x)
        x = self.bn1_s(x)
        x = self.relu(x)
        x = self.conv1_t(x)
        x = self.bn1_t(x)
        x = self.relu(x)

        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)

        # New
        for layer in self.layers:
            x = layer(x)
        # ...
        '''
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        '''

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ConvBlock3D(nn.Module):
    def __init__(self,args):
        super(ConvBlock3D,self).__init__(c_in,h_dim,kernel_size,stride = 1,padding=0,dilation=1,bias = True, device = 'cuda')
        self.conv3d = nn.Conv3d(c_in,h_dim,kernel_size,stride, padding,dilation,bias,device)
         
    def forward(self,inputs):
        return self.conv3d(inputs)
    
class CNN_VideoEncoder(nn.Module):
    def __init__(self,args,layers = [1, 1, 1, 1]):
        super(CNN_VideoEncoder,self).__init__()
        self.H_dims = args.netmob_H_dims
        self.c_in = args.netmob_c_in
        self.L = args.netmob_L
        self.z_dim = args.netmob_z_dim
        self.layers = layers

        #self.c_out = args.netmob_c_out
        #self.kernel_size = args.netmob_kernel_size
        #self.padding = args.netmob_padding
        #nb_blocks = len(self.H_dims)
        #blocks_dim_in =  [self.c_in] +self.H_dims[:-1]
        #blocks_dim_out = self.H_dims[1:] + [self.c_out] 

        # Define 3D Block to extract Feature from NetMob Video 
        block_inplanes = self.H_dims
        self.resnet_encoder =  ResNet(BasicBlock,self.layers, block_inplanes,
                    n_input_channels=self.c_in,
                    conv1_t_size=L, #7 ? 
                    conv1_t_stride=1,
                    no_max_pool=False,
                    shortcut_type='B',
                    widen_factor=1.0,
                    n_classes= self.z_dim
        )


    def forward(self,x,netmob_b):
        '''
        Captures the information contained in the NetMob image sequence associated to a single subway station

        Args 
        --------
        netmob_video_batch : a 5-th order tensor o size [B,C,H,W,L] 
        B : batch-size
        C : number of channel (i.e number of mobile-phone apps)
        H,W : height and width of image around the spatial unit
        L : length of historical sequence (t-w,t-d,t-6,t-5,t-4,t-3,t-2,t-1)


        Returns:
        --------
        outputs : a 2-th order tensor of size [B,Z]

        Z : latent space dimension 
        '''

        # [B,C,H,W,L]  ->  [B,C,L,H,W] 
        netmob_b = netmob_b.permute(0,1,4,2,3)
        
        # [B,L,C,H,W]   ->   [B,Z] 
        extracted_feature = self.resnet_encoder(netmob_b)







if __name__ == '__main__':

    # Hyperparameters
    c_out = 1
    H_dims = [64,128,256,512]
    kernel_size = (3,3,3)
    padding = (1,0,0) #Keep the temporal dimension L while reducing H,W 
    Z_dimension = 32

    # Init Inputs
    B,N,C,H,W,L = 32, 40, 4, 22,22,6
    netmob = torch.randn(B,N,C,H,W,L)
    netmob_station_i = netmob[:,0,:,:,:,:]

    # Define Args
    parser = argparse.ArgumentParser(description='Model Arguments')
    parser.add_argument(f'--netmob_c_in', type=int, default=C)
    parser.add_argument(f'--netmob_c_out', type=int, default=c_out)
    parser.add_argument(f'--netmob_H_dims', type=list, default=H_dims)
    parser.add_argument(f'--netmob_kernel_size', type=tuple, default=kernel_size)
    parser.add_argument(f'--netmob_padding', type=tuple, default=padding)
    parser.add_argument(f'--netmob_z_dim', type=int, default=Z_dimension)
    args = parser.parse_args(args=[])

    assert (len(H_dims)*(kernel_size[2]-1) < W) and (len(H_dims)*(kernel_size[1]-1) < H), ' too many 3D conv layer / Kernel-size to significant : inputs will be reduced to null tensor '

    encoder_model = CNN_VideoEncoder(args)
    extracted_feature = encoder_model(netmob_station_i)


    if True: 
        block = BasicBlock
        layers = [1, 1, 1, 1]  # nb 3D conv blocks par couches 
        block_inplanes = H_dims
        n_classes = Z_dimension
        ResNet(block,layers, block_inplanes,
                    n_input_channels=C,
                    conv1_t_size=L, #7 ? 
                    conv1_t_stride=1,
                    no_max_pool=False,
                    shortcut_type='B',
                    widen_factor=1.0,
                    n_classes=n_classes
        )

    
