import torch
import numpy as np

class Spatializer(objet):
    def __init__(self,H,W,L):
        super(Spatializer,self).__init__()

        sqrt = np.sqrt(L)
        if int(sqrt) == sqrt:
            self.H_new = sqrt*H
            self.W_new = sqrt*W
        else:
            self.H_new = (int(sqrt)+1)*H
            self.W_new = (int(sqrt)+1)*W 
    
    def forward(self,x):
        '''
        Inputs :
        --------
        6-th order tensor [B,N,c_in,H,W,L]  

        Outputs:
        --------
        4-th order tensor [B,out_dim]
        '''
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)  
        return(x)