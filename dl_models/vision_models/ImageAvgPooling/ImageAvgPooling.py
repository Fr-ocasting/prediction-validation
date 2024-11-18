import torch.nn as nn 

class ImageAvgPooling(nn.Module):
    def __init__(self):
        super(ImageAvgPooling,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self,x):
        '''
        Take as input a batch of Historical (temporal dim = L) image of NetMob (with channel C = number of apps)
        And return the average pooling on Channel and Width x Height dimensions.

        Is supposed to represent the information of the average intensity of presence around a spatial unit.

        inputs : 
        --------
        5-th order tensor [B,c_in,H,W,L]  

        outputs : 
        --------
        2-th order tensor [B,L]
        '''
        x = x.permute(0,4,1,2,3) # [B,C,H,W,L]  -> [B,L,C,H,W]
        x = self.avgpool(x)   # [B,L,C,H,W] -> [B,L,1,1,1]
        x = x.squeeze()   #  [B,L,1,1,1] -> [B,L]
        return(x)  

model = ImageAvgPooling