import argparse

parser = argparse.ArgumentParser(description='SpatialAttn')

parser.add_argument('--dim_model', type=int, default=32, choices=[16,32,64,128],  # 16
                    help='Number of embedding dimension of attn')

#parser.add_argument('--num_layers', type=int, default=2, choices=[1,2,3,4,5,6],  # 16
#                    help='Number of attn layer')

parser.add_argument('--num_heads', type=bool, default=2, 
                    choices = [1,2,3,4,8],
                    help='number of head for the multi-head attention.') 

parser.add_argument('--dim_feedforward', type=int, default=64, choices=[8,16,32,64,128,256,512],  #  16
                    help='Hidden Dim of the 2FC Layer module at output of each MHA')

parser.add_argument('--latent_dim', type=int, default=2, choices=[1,2,3,4,8,16],  #  16
                    help='Latent dimension of contextual channel dimension')

args = parser.parse_args(args=[])