from dl_models.vision_models.ResNet_2_1D import trivial_block_2PLus1D
import torch
import torch.nn as nn

class SimpleFeatureExtractor(nn.Module):
    def __init__(self,c_in,h_dim,out_dim):
        super(SimpleFeatureExtractor,self).__init__()

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



if __name__ == '__main__':
    from utilities_DL import forward_and_display_info

    B,N,C,H,W,L = 32, 40, 4, 22,22,6
    netmob = torch.randn(B,N,C,H,W,L)
    netmob_station_i = netmob[:,0,:,:,:,:]

    model = SimpleFeatureExtractor(C,128,256)
    output = forward_and_display_info(model,netmob_station_i)
