import torch.nn as nn

def conv_kxpxl(in_planes, mid_planes, k,p,l):
    padding_k = max(0,k-2)
    padding_p = max(0,p-2)
    padding_l = max(0,l-2)

    return nn.Conv3d(in_planes,
                     mid_planes,
                     kernel_size=(k,p,l),
                     stride=(1, 1, 1),
                     padding=(padding_k, padding_p, padding_l),
                     bias=False)
 
def conv3x1x1(in_planes, mid_planes):
    return(conv_kxpxl(in_planes, mid_planes, 3,1,1))

def conv1x1x3(in_planes, mid_planes):
    return(conv_kxpxl(in_planes, mid_planes, 1,1,3))

def conv1x3x3(in_planes, mid_planes):
    return(conv_kxpxl(in_planes, mid_planes, 1,3,3))

def conv3x3x1(in_planes, mid_planes):
    return(conv_kxpxl(in_planes, mid_planes, 3,3,1))

def conv1x1x1(in_planes, mid_planes):
    return(conv_kxpxl(in_planes, mid_planes, 1,1,1))