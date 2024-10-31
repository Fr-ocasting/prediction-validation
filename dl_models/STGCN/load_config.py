import argparse

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--Kt', type=int, default=3, choices=[2,3,4],
                    help='Kernel Size on the Temporal Dimension')

parser.add_argument('--stblock_num', type=int, default=2, choices=[2,3,4],
                    help='Number of STConv-blocks')

parser.add_argument('--Ks', type=int, default=2, choices=[1,2,3],
                    help='Number of iteration within the ChebGraphConv ONLY')

parser.add_argument('--graph_conv_type', type=str, default='graph_conv', 
                    choices = ['graph_conv','cheb_graph_conv'],
                    help='Type of graph convolution')

parser.add_argument('--gso_type', type=str, default='sym_norm_lap', 
                    choices = ['sym_norm_lap','rw_norm_lap','sym_renorm_adj','rw_renorm_adj'],
                    help='Type of calcul to compute the gso (Weighted Adjacency Matrix)')

parser.add_argument('--enable_bias', type=bool, default='True', 
                    choices = [True,False],
                    help='Enable bias on the output module (FC layers at the output of STGCN)')

parser.add_argument('--adj_type', type=str, default='corr', 
                    choices = ['adj','corr','dist'],
                    help='Type of Initial Adjacency Matrix (neighbors adjacency, distance correlation-based, or euclidian spatially-based distance)')

parser.add_argument('--enable_padding', type=bool, default=True,  
                    choices = [True,False],
                    help="Enable padding on the Temporal convolution. Suitable for short sequence cause (L' = L-2*(Kt-1)*stblock_num)")

parser.add_argument('--threeshold', type=float, default=0.3,  # between 0.0 and 1.
                    help="Threeshold to mask the Weighted Adjacency Matrix. < threeshold become 0")

parser.add_argument('--act_func', type=str, default='glu', 
                    choices = ['glu','gtu','silu'],
                    help="Type of activation function on the output module (FC layers at the output of STGCN)")

args = parser.parse_args(args=[])



