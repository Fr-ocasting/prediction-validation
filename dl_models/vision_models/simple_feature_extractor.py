import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
import math 
# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_file_path,'..','..'))
if grandparent_dir not in sys.path:
    sys.path.insert(0,grandparent_dir)
# ...

# Personnal import:
from dl_models.vision_models.ResNet_2_1D import trivial_block_2PLus1D

class ResBlock_2Plus1D(nn.Module):
    def __init__(self, c_in, c_out):
        super(ResBlock_2Plus1D, self).__init__()
        self.conv_block = trivial_block_2PLus1D(c_in, c_out)
        if c_in != c_out:
            self.downsample = nn.Conv3d(c_in, c_out, kernel_size=1,bias = False)
        else:
            self.downsample = None

        self.bn = nn.BatchNorm3d(c_out)
        self.relu = nn.ReLU()
    def forward(self, x):
        identity = x
        out = self.conv_block(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
            identity = self.bn(identity)
        out = out+ identity
        out = self.relu(out)
        return out

class FeatureExtractor_ResNetInspired_bis(nn.Module):
    def __init__(self, c_in=4, out_dim=64, N=40):
        super(FeatureExtractor_ResNetInspired_bis, self).__init__()
        self.n_vertex = n_vertex
        self.z_dim = out_dim

        # Calcul dynamique de N_h et N_w
        N_h = int(math.sqrt(N))
        N_w = (N + N_h - 1) // N_h  # Division entière supérieure pour couvrir tous les nœuds
        self.N_actual = N_h * N_w  # Nombre réel de nœuds après le pooling

        # Définition des couches
        self.init_avgpool = nn.MaxPool3d((2,2,1))
        self.init_conv = trivial_block_2PLus1D(c_in, 32)
        self.layer1 = ResBlock_2Plus1D(32, 64)
        self.layer2 = ResBlock_2Plus1D(64, 128)
        self.layer3 = ResBlock_2Plus1D(128, self.z_dim)
        self.avgpool = nn.AdaptiveAvgPool3d((N_h, N_w, 1))
        

    def forward(self, x):
        # x: [B, C, H, W, L]
        if (x.size(2) > 100)&(x.size(3) > 100):
            x = self.init_avgpool(x)  # reduce image dim
        x = self.init_conv(x)   # [B, 32, H, W, L]
        x = self.layer1(x)      # [B, 64, H, W, L]
        x = self.layer2(x)      # [B, 128, H, W, L]
        x = self.layer3(x)      # [B, z_dim, H, W, L]
        x = self.avgpool(x)     # [B, z_dim, N_h, N_w, 1]
        x = x.view(x.size(0), self.N_actual, self.z_dim)  # [B, N_actual, Z]
        x = x[:, :self.n_vertex, :]    # Sélection des N premiers nœuds si N_actual > N
        x = x.view(x.size(0), -1)  # [B, z_dim* N]
        return x 

class FeatureExtractor_ResNetInspired(nn.Module):
    def __init__(self,c_in,h_dim,L):
        super(FeatureExtractor_ResNetInspired,self).__init__()
        out_dim = L*h_dim//2

        self.init_avgpool = nn.MaxPool3d((2,2,1))
        self.block1 = trivial_block_2PLus1D(c_in,h_dim)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(2,2,1), padding=(0,0,1))
        self.block2 = trivial_block_2PLus1D(h_dim,out_dim)
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        
    
    def forward(self,x):
        '''
        Inputs :
        --------
        5-th order tensor [B,c_in,H,W,L]  

        Outputs:
        --------
        2-th order tensor [B,out_dim]
        '''
        if (x.size(2) > 100)&(x.size(3) > 100):
            x = self.init_avgpool(x)  # reduce image dim
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)  
        return(x)

class MinimalFeatureExtractor(nn.Module):
    def __init__(self,c_in,h_dim=16,L=8):
        super(MinimalFeatureExtractor,self).__init__()
        out_dim = L*h_dim//2

        self.relu = nn.ReLU()
        self.conv2d_1 = nn.Conv3d(c_in,h_dim,kernel_size=(1,3,3))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,2,2))

        self.conv2d_2 = nn.Conv3d(h_dim,2*h_dim,kernel_size=(1,3,3))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(2*L*h_dim,out_dim)


    def forward(self,x):
        '''
        Inputs :
        --------
        5-th order tensor [B,c_in,H,W,L]  

        Outputs:
        --------
        2-th order tensor [B,Z]=[B,L*h_dim//2]
        '''
        x = x.permute(0,1,4,2,3) # [B,C,H,W,L]  -> [B,C,L,H,W]

        # Feature extraction:
        x = self.conv2d_1(x)   #[B,C,L,H,W] -> [B,h_dim,L,H-2,W-2]
        if x.size(-1) > 8:
            x = self.maxpool1(x)   #[B,h_dim,L,H-2,W-2]-> [B,C,L,(H-5)/2,(W-5)/2] 
        x = self.conv2d_2(x)   #[B,h_dim,L,H'-4,W'-4] -> [B,2*h_dim,L,H'-6,W'-6]

        # Image Reduction 
        x = self.avgpool(x)   # [B,2*h_dim,L,H-6,W-6] -> [B,2*h_dim,L,1,1]
        x = x.view(x.size(0),-1)  # [B,2*h_dim,L,1,1] -> [B,2*L*h_dim]

        # Latent space reduction: 
        x = self.fc1(x)

        return(x)    


class ImageAvgPooling(nn.Module):
    def __init__(self):
        super(ImageAvgPooling,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self,x):
        '''
        5-th order tensor [B,c_in,H,W,L]   ->  2-th order tensor [B,L]
        '''
        x = x.permute(0,4,1,2,3) # [B,C,H,W,L]  -> [B,L,C,H,W]
        x = self.avgpool(x)   # [B,L,C,H,W] -> [B,L,1,1,1]
        x = x.squeeze()   #  [B,L,1,1,1] -> [B,L]
        return(x)  


if __name__ == '__main__':
    from utils.utilities_DL import forward_and_display_info
    from dl_models.vision_models.simple_feature_extractor import SimpleFeatureExtractor

    B,N,C,H,W,L = 32, 40, 4, 22,22,6
    netmob = torch.randn(B,N,C,H,W,L)
    netmob_station_i = netmob[:,0,:,:,:,:]

    model = SimpleFeatureExtractor(C,128,256)
    # model = TrivialImageReduction(C)
    output = forward_and_display_info(model,netmob_station_i)
