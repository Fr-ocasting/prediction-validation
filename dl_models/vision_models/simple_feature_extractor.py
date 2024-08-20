import torch
import torch.nn as nn

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


class FeatureExtractor_ResNetInspired(nn.Module):
    def __init__(self,c_in,h_dim,L):
        super(FeatureExtractor_ResNetInspired,self).__init__()
        out_dim = L*h_dim//2
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
        2-th order tensor [B,L]
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
