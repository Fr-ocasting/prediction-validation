import argparse

parser = argparse.ArgumentParser(description='CNN')

parser.add_argument('--c_in', type=int, default=1, 
                    help='Channel dimension of the input. Usually = 1. As we consider trafic forecasting, we could set c_in = 2 if two time series are considered : Speed and Flow')
parser.add_argument('--H_dims', type=list, default=[16,16], 
                    help="Hidden dimension of the two CNN layers. The number of Conv layers is as long as the list. The last element of the list must match the first elemnt of C_outs")
parser.add_argument('--C_outs', type=list, default=[16,1],  
                    help="Output dimension of the output layers (Sequence of 2FC layers). The number of FC layers is as long as the list. The channel dim of the output of the model is the last element of the list.")
parser.add_argument('--padding',type = int, default = 0,
                    help="Padding of the conv layer. Same for every conv layer.")
parser.add_argument('--kernel_size',type = int, default = (2,),
                    help="Size of the Conv2D kernel.")
args = parser.parse_args(args=[])
